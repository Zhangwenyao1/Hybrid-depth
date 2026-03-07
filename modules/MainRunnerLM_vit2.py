# MainRunnerLM.py
# Contains definition of the pytorch LightningModule used in this work.
# Actual model definitions (nn.Module) are in their own files.

import os, sys, logging
import torch
import torchvision
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from datasets.dataloader import DepthDataLoader # Original Adabins dataloader
from datasets.dataloader import LaneDepthMixDataLoader
from modules.Preprocess import Preprocess
from modules.DataAugmentation import DataAugmentation
from modules.AdaBins import AdaBins
from modules.DepthCLIP_dino import DepthCLIP
# from datasets.data import RegressionDataModule

from losses.LossWrapper import LossWrapper
from metrics.MetricsPreprocess import MetricsPreprocess
from metrics.AbsRel import AbsRel, AbsRelRunningAvg
from metrics.SqRel import SqRel, SqRelRunningAvg
from metrics.RMSE import RMSE, RMSERunningAvg
from metrics.RMSELog import RMSELog, RMSELogRunningAvg
from metrics.AccThresh import AccThresh, AccThreshRunningAvg
from metrics.Log10 import Log10, Log10RunningAvg

from figurebuilders.FigureBuilder import FigureBuilder
from figurebuilders.FigureBuilderTusimple import FigureBuilderTusimple

import matplotlib
from matplotlib import pyplot as plt
from .optim import build_lr_scheduler, build_optimizer, build_staged_lr_param_groups

class MainRunnerLM(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        # Data preprocessing/augmentation/normalization modules

            
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    # ImageNet statistics
        self.unnormalize = torchvision.transforms.Compose([
                                torchvision.transforms.Normalize(mean = [ 0.0, 0.0, 0.0 ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                torchvision.transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1.0, 1.0, 1.0 ]),
                               ])
        
        # Metrics and metrics preprocessing
        self.metrics_preprocess = MetricsPreprocess(self.args)

        # For making neat grids of batches + predictions
        if self.args.model.name == "adabins" or self.args.model.name == "depthclip":
                self.figure_builder = FigureBuilder(self.args, num_samples=min(self.args.basic.batch_size, 4)) # Up to 4 samples from the batch
                self.figure_builder_patch = FigureBuilderTusimple(self.args, num_samples=min(self.args.basic.batch_size, 4))
        else:
                sys.exit("Error: not implemented")

        # Example input for use when building model graph
        if self.args[self.args.basic.dataset].do_kb_crop:
            # 376, 1241
            eg_height = 352
            eg_width = 1216
        else:
            eg_height = self.args[self.args.basic.dataset].dimensions_train[0]
            eg_width = self.args[self.args.basic.dataset].dimensions_train[1]
        # self.example_input_array = {
        #     'image': torch.rand((self.args.basic.batch_size, 3, eg_height, eg_width)),
        #     'depth': torch.rand((self.args.basic.batch_size, 1, eg_height, eg_width)),
        # }
        
        row_example = torch.tensor([-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 
                567, 532, 496, 461, 425, 390, 355, 319, 284, 
                248, 213, 177, 142, 106, 71, 35, -2, -2, -2, 
                -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 
                -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
                -2, -2, -2, -2])
        column_example = torch.tensor(
            [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 
             280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 
             400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 
             520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 
             640, 650, 660, 670, 680, 690, 700, 710
        ])
        
        self.example_input_array = (
            torch.rand((1, 3, eg_height, eg_width)),
            column_example.unsqueeze(0),
            row_example.unsqueeze(0).unsqueeze(0))
        

        self.var_acc = 0.0
        self.var_acc_counter = 0

        self.most_recent_train_batch = None   # To be used to store the most recent batch.
        self.most_recent_val_batch = None   # To be used to store the most recent batch.

        # Model and loss definitions
        if self.args.model.name == 'adabins':
                self.model = AdaBins(self.args)
        elif self.args.model.name == "depthclip":
                self.model = DepthCLIP(self.args)
        else:
                sys.exit(f"Error: unrecognised model ({self.args.model.get('name')})")
        for p in self.model.get_frozen_params():
            p.requires_grad = False

        name_list = []
        non_name_list = []
        for n,p in self.model.named_parameters():
            if p.requires_grad:
                name_list.append(n)
            else:
                non_name_list.append(n)
        
                
        print('trainable parameters',name_list)  
        
        # For prediction/inference mode: a dict for storing per-example metrics, paths, filenames.
        self.prediction_dict = {}
        self.loss = LossWrapper(self.args)
        self.rank_loss = nn.MarginRankingLoss()
        # Metric definitions
        self.abs_rel = AbsRel(self.args)
        self.sq_rel = SqRel(self.args)
        self.rmse = RMSE(self.args)
        self.rmse_log = RMSELog(self.args)
        self.log10 = Log10(self.args)
        self.acc_1 = AccThresh(self.args, threshold=1.25)
        self.acc_2 = AccThresh(self.args, threshold=1.25 ** 2)
        self.acc_3 = AccThresh(self.args, threshold=1.25 ** 3)

        self.abs_rel_ra = AbsRelRunningAvg(self.args)
        self.sq_rel_ra = SqRelRunningAvg(self.args)
        self.rmse_ra = RMSERunningAvg(self.args)
        self.rmse_log_ra = RMSELogRunningAvg(self.args)
        self.log10_ra = Log10RunningAvg(self.args)
        self.acc_1_ra = AccThreshRunningAvg(self.args, threshold=1.25)
        self.acc_2_ra = AccThreshRunningAvg(self.args, threshold=1.25 ** 2)
        self.acc_3_ra = AccThreshRunningAvg(self.args, threshold=1.25 ** 3)
        # self.tusimple_data = RegressionDataModule(self.args)

    def forward(self, *batch, **kwargs):
        """Forward method.
        
        Args:
            batch (tuple of image, depth): the input image and corresponding ground truth.
        """
        image, height, width = batch
        # image, depth_gt = batch["image"], batch["depth"]

        return self.model(image, height, width)

    def transform_convert(self, img_tensor, transform):
        """
        param img_tensor: tensor
        param transforms: torchvision.transforms
        """
        from torchvision  import transforms
        if 'Normalize' in str(transform):
            normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
            mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
            std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
            img_tensor.mul_(std[:,None,None]).add_(mean[:,None,None])
            
        img_tensor = img_tensor.transpose(0,2).transpose(0,1)  # C x H x W  ---> H x W x C
        
        if 'ToTensor' in str(transform) or img_tensor.max() < 1:
            img_tensor = img_tensor.detach().cpu().numpy()*255
        
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.numpy()
        
        if img_tensor.shape[2] == 3:
            # img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
            img = img_tensor.astype('uint8')
        elif img_tensor.shape[2] == 1:
            # img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
            img = img_tensor.astype('uint8')
        else:
            raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))
            
        return img
    
    def training_step(self, batch, batch_idx):
        # image, depth_gt = batch["image"], batch["depth"]

        if self.args.basic.dataset == 'tusimple':
             
            image, heights, width, img_shape_list, resize_ratios = batch["image"], batch["height"], batch["width"], batch['input_shapes'].tolist(), batch['resize_ratios']
            # output_named_tuple = self.model(image, heights, weights)
            logits, contrastive_loss, depths_pred, selected_patches_coords = self.model(image, heights, width, img_shape_list, resize_ratios)
            rank_loss = 0
            for logits_per_img in logits:
                for item in logits_per_img:
                    rank_loss += self.compute_losses(item, 'train', self.args.loss.get('use_rank_biloss', False))['rank_loss']
            if len(logits) > 0:
                rank_loss /= len(logits)
            # rank_loss = 0
            depth_loss = 0.0
            kitti_mask = torch.Tensor([True if depth_gt is not None else False for depth_gt in batch['depths']]).bool()
            depths_gt = [batch['depths'][i] for i in range(len(batch['depths'])) if batch['depths'][i] is not None]
            for i in range(len(depths_gt)):
                depth_gt = depths_gt[i]
                depth_pred_interp = nn.functional.interpolate(depths_pred[i].unsqueeze(0), depth_gt.shape[-2:], mode='bilinear', align_corners=True)
                depth_loss += self.loss(depth_pred_interp, depth_gt[None, None, :, :], None, None)
            if kitti_mask.sum() > 0:
                depth_loss /= kitti_mask.sum()

            if not self.args.loss.get('use_rank_loss', True):
                rank_loss = 0.0 * rank_loss
            if not self.args.loss.get('use_contrastive_loss', True):
                contrastive_loss = 0.0 * contrastive_loss
            if not self.args.loss.get('use_depth_loss', True):
                depth_loss = 0.0 * depth_loss
            loss = rank_loss + contrastive_loss + depth_loss

            self.most_recent_train_batch = {
                'kitti_mask': kitti_mask,
                'similarity_map': logits,
                'resize_ratios': resize_ratios,
                'sampled_patches': selected_patches_coords,
                'image': image,
                'depth_gt': depths_gt,
                'depth_pred': [depths_pred[i].squeeze() for i in range(len(depths_gt))],
                'contrastive_loss':contrastive_loss,
                'rank_loss': loss
            }
            self.log_dict(dict(rank_loss=rank_loss, contrastive_loss=contrastive_loss, depth_loss=depth_loss), on_step=True, prog_bar=True)
            # with open(f'/output/{self.global_step}.txt', 'w') as f:
            #     lines = []
            #     for img_file in batch['img_files']:
            #         lines.append(img_file+'\n')
            #     f.writelines(lines)
            #self.log("train/loss", loss, on_step=True, prog_bar=True) 
            #self.log("Train, Depth loss: {}, Rank loss: {}, Total loss: {}".format(depth_loss, rank_loss, loss), on_step=True, prog_bar=True)
            if self.global_step % (1 if self.args.debug else 500) == 0:
                if isinstance(self.logger, TensorBoardLogger):
                    self.logger.experiment.add_figure(tag="train/samples_kitti", figure=self.figure_builder.build(self.most_recent_train_batch), global_step=self.global_step)
                    self.figure_builder.reset()        
                    self.logger.experiment.add_figure(tag="train/samples_tusimple", figure=self.figure_builder_patch.build(self.most_recent_train_batch), global_step=self.global_step)
                    self.figure_builder_patch.reset()  
                elif isinstance(self.logger, WandbLogger):
                    self.logger.log_image(key="samples_kitti", images=[wandb.Image(self.figure_builder.build(self.most_recent_train_batch))], caption=['samples_kitti'], step=self.global_step)
                    self.figure_builder.reset()
                    self.logger.log_image(key="samples_tusimple", images=[wandb.Image(self.figure_builder_patch.build(self.most_recent_train_batch))], caption=['samples_tusimple'], step=self.global_step)
                    self.figure_builder_patch.reset()  
            return loss
        else: 
            image = batch["image"]
            output_named_tuple = self.model(image)
            depth_pred = output_named_tuple.depth_pred  # No clamping is used during training, but min/max clamping and nan/inf removal are used during eval/val
            
            # depth_pred = depth_pred.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # depth_gt = depth_gt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            depth_mask = (depth_gt > self.args[self.args.basic.dataset].min_depth) # BTS and AdaBins only use a minimum mask during training, but use both during validation/evaluation

            loss = self.loss(depth_pred=depth_pred, depth_gt=depth_gt, depth_mask=depth_mask, output_named_tuple=output_named_tuple)

            self.most_recent_train_batch = {
                'image': image,
                'depth_gt': depth_gt,
                'depth_pred': nn.functional.interpolate(depth_pred, depth_gt.shape[-2:], mode='bilinear', align_corners=True),
                'loss': loss
            }
            self.log("train/loss", loss, on_step=True, prog_bar=True) 

            return loss


    # def on_train_epoch_end(self):#, training_step_outputs):
    #     self.logger.experiment.add_figure(tag="train/samples_kitti", figure=self.figure_builder.build(self.most_recent_train_batch), global_step=self.global_step)
    #     self.figure_builder.reset()        
    #     self.logger.experiment.add_figure(tag="train/samples_tusimple", figure=self.figure_builder_patch.build(self.most_recent_train_batch), global_step=self.global_step)
    #     self.figure_builder_patch.reset()  
    

    def validation_step(self, batch, batch_idx):
        image, heights, width = batch["image"], batch["height"], batch["width"]
        # image, depth_gt = batch["image"], batch["depth"]
        # image =  batch["image"]
        # Validation run on image and its mirror, then averaged, following previous work.
        # Regular predictions:
        if self.args.basic.dataset == 'tusimple':
            logits,contrastive_loss = self.model(image, heights, width)
            rank_loss = 0
            for item in logits:
                rank_loss += self.compute_losses(item, 'train')['rank_loss']
            # rank_loss = 0
            # depth_loss = 0
            loss = rank_loss + contrastive_loss

            self.most_recent_train_batch = {
                'image': image,
                'loss': loss
            }
            self.log("train/loss", loss, on_step=True) 
            # self.log("Train, Depth loss: {}, Rank loss: {}, Total loss: {}".format(depth_loss, rank_loss, loss), on_step=True)
            return loss
        else:
            depth_gt = batch["depth"]
            depth_mask = (depth_gt > self.args[self.args.basic.dataset].min_depth) & (depth_gt <= self.args[self.args.basic.dataset].max_depth)
            loss = self.loss(depth_pred=depth_pred_final, depth_gt=depth_gt, depth_mask=depth_mask, output_named_tuple=output_named_tuple)

            self.most_recent_val_batch = {
                'image': image,
                'depth_gt': depth_gt.clone(),
                'depth_pred': nn.functional.interpolate(depth_pred_final.clone(), depth_gt.shape[-2:], mode='bilinear', align_corners=True),
                'loss': loss
            }
            depth_pred_m, depth_mask_m = self.metrics_preprocess(depth_pred=depth_pred_final.clone(), depth_gt=depth_gt.clone())
            # Apply any crops (eigen/garg) if necessary (metrics preprocessing), then apply validity mask.
        
            depth_pred_m, depth_gt_m = depth_pred_m[depth_mask_m], depth_gt[depth_mask_m]
        
            # Compute metrics
            self.abs_rel(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
            self.sq_rel(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
            self.rmse(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
            self.rmse_log(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
            self.log10(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
            self.acc_1(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
            self.acc_2(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
            self.acc_3(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())

            self.abs_rel_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
            self.sq_rel_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
            self.rmse_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
            self.rmse_log_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
            self.log10_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
            self.acc_1_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
            self.acc_2_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
            self.acc_3_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())

            # Log metrics and loss
            # self.log("metrics/abs_rel", self.abs_rel, on_epoch=True)
            self.log("metrics/sq_rel", self.sq_rel, on_epoch=True)
            self.log("metrics/rmse", self.rmse, on_epoch=True)
            self.log("metrics/rmse_log", self.rmse_log, on_epoch=True)
            self.log("metrics/log10", self.log10, on_epoch=True)
            self.log("metrics/acc_1", self.acc_1, on_epoch=True)
            self.log("metrics/acc_2", self.acc_2, on_epoch=True)
            self.log("metrics/acc_3", self.acc_3, on_epoch=True)

            self.log("metrics_ra/abs_rel_ra", self.abs_rel_ra, on_epoch=True)
            self.log("metrics_ra/sq_rel_ra", self.sq_rel_ra, on_epoch=True)
            self.log("metrics_ra/rmse_ra", self.rmse_ra, on_epoch=True)
            self.log("metrics_ra/rmse_log_ra", self.rmse_log_ra, on_epoch=True)
            self.log("metrics_ra/log10_ra", self.log10_ra, on_epoch=True)
            self.log("metrics_ra/acc_1_ra", self.acc_1_ra, on_epoch=True)
            self.log("metrics_ra/acc_2_ra", self.acc_2_ra, on_epoch=True)
            self.log("metrics_ra/acc_3_ra", self.acc_3_ra, on_epoch=True)

            self.log("val/loss", loss, on_epoch=True, sync_dist=True)

            return loss


    def on_validation_epoch_end(self, validation_step_outputs):
        # This dict is used at the end of training to get a plaintext, easily copy-pastable version of the most recent metrics.
        if self.args.basic.dataset=='tusimple':
            self.last_metrics_dict = {
                "abs_rel": self.abs_rel.compute(),
                "sq_rel": self.sq_rel.compute(),
                "rmse": self.rmse.compute(),
                "rmse_log": self.rmse_log.compute(),
                "log10": self.log10.compute(),
                "acc_1": self.acc_1.compute(),
                "acc_2": self.acc_2.compute(),
                "acc_3": self.acc_3.compute(),
            }
        else:           
            self.last_metrics_dict = {
                "abs_rel": self.abs_rel.compute(),
                "sq_rel": self.sq_rel.compute(),
                "rmse": self.rmse.compute(),
                "rmse_log": self.rmse_log.compute(),
                "log10": self.log10.compute(),
                "acc_1": self.acc_1.compute(),
                "acc_2": self.acc_2.compute(),
                "acc_3": self.acc_3.compute(),
                "abs_rel_ra": self.abs_rel_ra.compute(),
                "sq_rel_ra": self.sq_rel_ra.compute(),
                "rmse_ra": self.rmse_ra.compute(),
                "rmse_log_ra": self.rmse_log_ra.compute(),
                "log10_ra": self.log10_ra.compute(),
                "acc_1_ra": self.acc_1_ra.compute(),
                "acc_2_ra": self.acc_2_ra.compute(),
                "acc_3_ra": self.acc_3_ra.compute(),
            }
            self.logger.experiment.add_figure(tag="val/samples", figure=self.figure_builder.build(self.most_recent_val_batch), global_step=self.global_step)
            self.figure_builder.reset()

            print("LOGIT VARIANCE: ")
            print(self.var_acc / self.var_acc_counter)

    
    def on_train_end(self):
        pass
        # log_str = f"abs_rel, sq_rel, rms, rmsl, log10, d1, d2, d3:  \n {self.last_metrics_dict['abs_rel']}, {self.last_metrics_dict['sq_rel']}, {self.last_metrics_dict['rmse']}, {self.last_metrics_dict['rmse_log']}, {self.last_metrics_dict['log10']}, {self.last_metrics_dict['acc_1']}, {self.last_metrics_dict['acc_2']}, {self.last_metrics_dict['acc_3']}  \n ==#==  \nabs_rel_ra, sq_rel_ra, rms_ra, rmsl_ra, log10_ra, d1_ra, d2_ra, d3_ra:  \n{self.last_metrics_dict['abs_rel_ra']}, {self.last_metrics_dict['sq_rel_ra']}, {self.last_metrics_dict['rmse_ra']}, {self.last_metrics_dict['rmse_log_ra']}, {self.last_metrics_dict['log10_ra']}, {self.last_metrics_dict['acc_1_ra']}, {self.last_metrics_dict['acc_2_ra']}, {self.last_metrics_dict['acc_3_ra']}"
        # self.logger.experiment.add_text("metrics/all", log_str, global_step=self.global_step)
        # hparam_dict = {
        #     "batch size": self.args.basic.batch_size,
        #     "use_swa": ("use_swa" in self.args.optimizer and self.args.optimizer.use_swa),
        #     "model": self.args.model.name,
        #     "encoder name": self.args[self.args.model.name].get("encoder_name"),
        #     "clip": self.args[self.args.model.name].get("clip"),
        #     "current epoch": self.current_epoch,
        #     "precision": self.precision,
        # }
        
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """ Prediction step will take a prediction (without test-time augmentation), run evaluation on it as well, and
        save the whole batch (input and outputs, including any auxiliary outputs) plus the metrics to disk. 
        Whatever gets returned here ends up in a list (handled by lightning), and that list gets returned when calling
        trainer.predict() with a dataloader.
        """

        # image, depth_gt = batch["image"], batch["depth"]
        image, heights, width = batch["image"], batch["height"], batch["width"]

        # Regular predictions:
        output_named_tuple, contrastive_loss = self.model(image, heights, width)
        depth_pred = output_named_tuple.depth_pred
        depth_pred = torch.clamp(
            depth_pred,
            min=self.args[self.args.basic.dataset].min_depth,
            max=self.args[self.args.basic.dataset].max_depth
        )
        # nan and inf handling done in metrics_preprocess

        depth_mask = (depth_gt > self.args[self.args.basic.dataset].min_depth) & (depth_gt <= self.args[self.args.basic.dataset].max_depth)

        loss = self.loss(depth_pred=depth_pred, depth_gt=depth_gt, depth_mask=depth_mask, output_named_tuple=output_named_tuple)

        self.most_recent_pred_batch = {
            'image': image,
            'depth_gt': depth_gt.clone(),
            'depth_pred': nn.functional.interpolate(depth_pred.clone(), depth_gt.shape[-2:], mode='bilinear', align_corners=True),
            'loss': loss
        }

        depth_pred = nn.functional.interpolate(depth_pred.clone(), depth_gt.shape[-2:], mode='bilinear', align_corners=True)


        # Apply any crops (eigen/garg) if necessary (metrics preprocessing), then apply validity mask.
        depth_pred_m, depth_mask_m = self.metrics_preprocess(depth_pred=depth_pred.clone(), depth_gt=depth_gt.clone())
        depth_pred_m, depth_gt_m = depth_pred_m[depth_mask_m], depth_gt[depth_mask_m]
        
        # Compute metrics
        self.abs_rel(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.sq_rel(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.rmse(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.rmse_log(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.log10(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.acc_1(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.acc_2(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.acc_3(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())

        self.abs_rel_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.sq_rel_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.rmse_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.rmse_log_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.log10_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.acc_1_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.acc_2_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())
        self.acc_3_ra(depth_pred=depth_pred_m.clone(), depth_gt=depth_gt_m.clone())

        # Turn depth maps into RGB
        if self.args.get("inf_save_mode") == 'all' or self.args.get("inf_save_mode") == 'some':
                if self.args.get("inf_save_mode") == "all":
                    save_mod = 1
                elif self.args.get("inf_save_mode") == "some":
                    save_mod = 100
                
                if batch_idx % save_mod == 0:
                    cmap_gt = matplotlib.cm.get_cmap("inferno_r")
                    cmap_gt.set_bad(color='1')
                    cmap_gt.set_under(color='1')
                    cmap_pred = matplotlib.cm.get_cmap("inferno_r")

                    depth_min = self.args[self.args.basic.dataset].min_depth
                    # breakpoint()
                    depth_max = depth_gt.max()

                    # Save things, using batch_idx as base name
                    normed_image = self.unnormalize(batch['image'][0]).cpu()

                    plt.clf()
                    plt.axis('off')
                    plt.imshow(np.transpose(normed_image.detach().cpu().numpy(), (1, 2, 0)))
                    plt.savefig(os.path.join(self.args.predict_output_dir, f"{batch_idx}_im.png"), bbox_inches='tight', dpi=250)
                    
                    # Depth visualisations
                    plt.imshow(depth_gt[0].detach().cpu().numpy().squeeze(0), vmin=depth_min, vmax=depth_max, cmap=cmap_gt)
                    plt.savefig(os.path.join(self.args.predict_output_dir, f"{batch_idx}_depth_gt.png"), bbox_inches='tight', dpi=250)
                    plt.imshow(depth_pred[0].detach().cpu().numpy().squeeze(0), vmin=depth_min, vmax=depth_max, cmap=cmap_pred)
                    plt.savefig(os.path.join(self.args.predict_output_dir, f"{batch_idx}_depth_pred.png"), bbox_inches='tight', dpi=250)
                    
                    # Depth raw values
                    # torch.save(depth_gt[0].detach().cpu(), os.path.join(self.args.predict_output_dir, f"{batch_idx}_depth_gt_raw.pkl"))
                    # torch.save(depth_pred[0].detach().cpu(), os.path.join(self.args.predict_output_dir, f"{batch_idx}_depth_pred_raw.pkl"))
            
        else:
                pass    # For readability. 

        # Make a dict of all the metrics + filenames, then add to the prediction dict for later saving
        curr_pred_dict = {
            "batch_idx": batch_idx,
            "image_filename": batch["image_path"][0],
            "depth_gt_filename": batch["depth_path"][0],

            "abs_rel": self.abs_rel.compute().item(),
            "sq_rel": self.sq_rel.compute().item(),
            "rmse": self.rmse.compute().item(),
            "rmse_log": self.rmse_log.compute().item(),
            "log10": self.log10.compute().item(),
            "acc_1": self.acc_1.compute().item(),
            "acc_2": self.acc_2.compute().item(),
            "acc_3": self.acc_3.compute().item(),

            "abs_rel_ra": self.abs_rel_ra.compute().item(),
            "sq_rel_ra": self.sq_rel_ra.compute().item(),
            "rmse_ra": self.rmse_ra.compute().item(),
            "rmse_log_ra": self.rmse_log_ra.compute().item(),
            "log10_ra": self.log10_ra.compute().item(),
            "acc_1_ra": self.acc_1_ra.compute().item(),
            "acc_2_ra": self.acc_2_ra.compute().item(),
            "acc_3_ra": self.acc_3_ra.compute().item(),
            "loss": loss.item(),
        }

        self.prediction_dict[batch_idx] = curr_pred_dict

        # Manually reset every batch as we want to save individual metrics, not batch metrics.
        self.abs_rel.reset()
        self.sq_rel.reset()
        self.rmse.reset()
        self.rmse_log.reset()
        self.log10.reset()
        self.acc_1.reset()
        self.acc_2.reset()
        self.acc_3.reset()

        self.abs_rel_ra.reset()
        self.sq_rel_ra.reset()
        self.rmse_ra.reset()
        self.rmse_log_ra.reset()
        self.log10_ra.reset()
        self.acc_1_ra.reset()
        self.acc_2_ra.reset()
        self.acc_3_ra.reset()

        return loss


    def on_predict_end(self):
        # Save the prediction dict to file
        out_df = pd.DataFrame.from_dict(self.prediction_dict, orient='index')
        out_path = os.path.join(self.args.predict_output_dir, f"prediction_metrics.csv")    
        out_df.to_csv(out_path)


    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Perform data augmentation on GPU (if doing training), and normalize.
        
        Expects batch to be a dict containing keys 'image' and 'depth_gt'.
        Returns a tuple of (image, depth).
        """

        if self.args.basic.get("use_adabins_dataloader") != True:
            # The adabins dataloader takes care of both normalization and data augmentation, so we only do those things if not using it
            if self.trainer.training:
                batch = self.data_augmentation(batch)

            batch['image'] = self.normalize(batch['image']) # Important: Normalise images to ImageNet mean and std.
        
            return batch
        else:
            # If using the adabins dataloader, the keys are different.
            if self.training:
                return batch
            else:
                # print(batch['depth']) 
                if self.args.basic.dataset=='tusimple':
                    # batch['depth'] = batch['depth']
                    # batch['image'] = batch['image']
                    image, heights, width = batch["image"], batch["height"], batch["width"]
                else:
                    batch['depth'] = batch['depth'].permute(0, 3, 1, 2).contiguous()
                    
                return batch

    def compute_losses(self, logits, gather_type='train', biloss=False):
        losses = {}

        z = torch.zeros(1).cuda()
        losses["rank_loss"] = z
        if gather_type == 'train':
            y1 = torch.ones(1).cuda()
            for i in range(logits.size()[0]):
                for j in range(i+1, logits.size()[0]):
                    losses["rank_loss"] += self.rank_loss(logits[j][j].unsqueeze(0), logits[i][j].unsqueeze(0), y1)
            if biloss:
                for i in range(logits.size()[0]-1, 0, -1):
                    for j in range(0, i):
                        losses["rank_loss"] += self.rank_loss(logits[j][j].unsqueeze(0), logits[i][j].unsqueeze(0), y1)
        return losses
    
    
    def configure_optimizers(self):
        if "slow_encoder" in self.args[self.args.model.name]:
            params = [
                {"params": self.model.get_encoder_params(), "lr": self.args.optimizer.lr / self.args[self.args.model.name].slow_encoder},
                {"params": self.model.get_non_encoder_params(), "lr": self.args.optimizer.lr},
                # {"params": self.model.get_zero_params(), "lr": 0.0}
            ]
        else:
            params = [
                {"params": self.model.get_encoder_params(), "lr": self.args.optimizer.lr},
                {"params": self.model.get_non_encoder_params(), "lr": self.args.optimizer.lr},
                # {"params": self.model.get_zero_params(), "lr": 0.0}
            ]

        # params = [{"params": self.model.get_encoder_params(), "lr": self.args.optimizer.lr}]

        # Freezing anything that we want frozen
        # Note that if grads need to propagate through then params should have requires_grad=True, but not be added to an optimizer (so they don't update).
        for p in self.model.get_frozen_params():
            p.requires_grad = False
        for p in self.model.get_encoder_params():
            p.requires_grad = True
        optimizer = torch.optim.AdamW(params=params, lr=self.args.optimizer.lr, weight_decay=self.args.optimizer.wd)        
        # optimizer = torch.optim.SGD(params=params, lr=self.args.optimizer.lr, weight_decay=self.args.optimizer.wd, momentum=0.9)
        
        if "use_swa" not in self.args.optimizer or ("use_swa" in self.args.optimizer and self.args.optimizer.use_swa):
            # self.trainer.estimated_stepping_batches = 10000
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                            max_lr=self.args.optimizer.lr,
                            total_steps=self.trainer.estimated_stepping_batches,
                            cycle_momentum=True,
                            base_momentum=0.85, max_momentum=0.95, last_epoch=-1,
                            div_factor=self.args.optimizer.div_factor,
                            final_div_factor=self.args.optimizer.final_div_factor)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }
        # else:
        return optimizer


    #     return self.build_optmizer_and_scheduler(**self._optimizer_and_scheduler_cfg)

    # def build_optmizer_and_scheduler(
    #     self,
    #     param_dict_cfg=None,
    #     optimizer_cfg=None,
    #     lr_scheduler_cfg=None,
    # ):
        # print('param_dict_cfg:', param_dict_cfg)
        # param_dict_ls = self.build_param_dict(**param_dict_cfg)

        # optim = build_optimizer(
        #     model=params,
        #     **self.args.optimizer_cfg,
        # )
        # sched = build_lr_scheduler(optimizer=optim, **self.args.lr_scheduler_cfg)
        # return [optim], [sched]



    
    def train_dataloader(self):

        #train_loader = DepthDataLoader(self.args, mode='train').data
        train_loader = LaneDepthMixDataLoader(self.args, mode='train').data
        return train_loader


    def val_dataloader(self):
        val_loader = DepthDataLoader(self.args,mode='online_eval').data
        return val_loader


    def predict_dataloader(self):

        predict_loader = DepthDataLoader(self.args,mode='online_eval').data

        return predict_loader
        