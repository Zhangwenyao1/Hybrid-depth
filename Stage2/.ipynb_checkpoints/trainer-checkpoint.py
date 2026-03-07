# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time
import warnings
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from collections import OrderedDict
import json
import wandb
import networks.depthclip_pose_dinocls
import networks.dino_clip_encoder
from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed
from omegaconf import OmegaConf

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        # print(self.log_path)
        # exit(0)
        # checking height and width are multiples of 32
        if self.opt.height % 32 != 0:
            warnings.warn("'height' shoule be a multiple of 32")
        if self.opt.width % 32 != 0:
            warnings.warn("'width' shoule be a multiple of 32")

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")


        if self.opt.depth_model_type == 'depthclip_ed':     
            #self.models["encoder"] = networks.depthclipencoder(self.opt)
            self.models['encoder'] = networks.depthdinoclipencoder(self.opt)
            self.models["encoder"].to(self.device)
            use_clip = not self.opt.only_dino
            if use_clip:
                if not self.opt.only_clip:
                    dpt_in_channels = 1024
                else:
                    dpt_in_channels = 256
            else:
                dpt_in_channels = 768

            if self.opt.use_depth_text_align:
                if self.opt.cat_depth_text_logic:
                    dpt_in_channels += self.models['encoder'].lenth
                else:
                    dpt_in_channels = self.models['encoder'].lenth

            self.models["depth"] = networks.depthclipdecoder(self.opt, in_channels=dpt_in_channels, out_channels=[256, 512, 1024, 1024])
            # self.models["depth"] = networks.MSDepthDecoder(
            #     np.array([64, 256, 512, 1024, 2048]), self.opt.scales, discretization=self.opt.discretization)
            # self.models["depth"] = networks.HRDepthDecoder(np.array([64, 256, 512, 1024, 2048]))
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["depth"].parameters())
            
            # print("load the pretrained weights")
            # pretrained_path = '/code/occ/PromptLearningCLIP-MDE/runs/basicParams/version_23/checkpoints/epoch=0-step=13741.ckpt'    
            # pretrained_path = '/code/occ/PromptLearningCLIP-MDE/runs/basicParams/version_73/checkpoints/epoch=0-step=13741.ckpt'
            # pretrained_path = '/code/occ/PromptLearningCLIP-MDE/runs/basicParams/version_144/encoder.ckpt' 
            # state_dict = torch.load(pretrained_path)['state_dict']
            # new_state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}      
            # self.models['encoder'].load_state_dict(new_state_dict, strict=False)
            # self.models['depth'].load_state_dict(new_state_dict, strict=False)

            # lhs modified
            pretrained_path = self.opt.stage1_checkpoint_path
            if pretrained_path != "":
                print('-------------- load stage1 checkpoint -----------')
                state_dict = torch.load(pretrained_path)['state_dict']
                new_state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}  
                self.models['encoder'].load_state_dict(new_state_dict, strict=False)
                    
            self.sgd_name_list = []
            self.sgd_param_list = []
            self.adam_name_list = []
            self.adam_param_list = []
            names = []
            for n,p in self.models['encoder'].named_parameters():
                if 'clip' in n:
                    if 'clip.transformer' in n:
                        p.requires_grad_(False)
                    else:
                        p.requires_grad_(True)
                elif 'dino' in n:
                    if "dinov2.blocks.11" in n:
                        p.requires_grad_(True)
                    else:
                        p.requires_grad_(False)
                elif 'extra_tkns' in n:
                    p.requires_grad_(True)
                elif 'resnet50' in n:
                    p.requires_grad_(True)
                # elif "clip_align_conv" in n:
                #     p.requires_grad_(True)
                else:
                    p.requires_grad_(False)
                    # raise NotImplementedError
            # for n,p in self.models['encoder'].named_parameters():
            #     if 'visual' in n:
            #         if 'lora' in n:
            #             p.requires_grad_(True)
            #         else:
            #             p.requires_grad_(False)
            #     else:
            #         p.requires_grad_(False)
                # if 'clip.transformer' in n:
                #     p.requires_grad_(False)
                # elif 'logit_scale' in n:
                #     p.requires_grad_(False)
                #     names.append(n)
                # elif 'clip.text_projection' in n:
                #     p.requires_grad_(False)
                #     names.append(n)
                # elif 'clip.positional_embedding' in n:
                #     p.requires_grad_(False)
                #     names.append(n)
                # elif 'clip.token_embedding' in n:
                #     p.requires_grad_(False)
                #     names.append(n)
                # elif 'clip.ln_final' in n:
                #     p.requires_grad_(False)
                #     names.append(n)
                # elif 'clip.visual' in n:
                #     p.requires_grad_(True)
                #     self.adam_name_list.append(n)
                #     self.adam_param_list.append(p)
                #     names.append(n)
                # else:
                #     p.requires_grad_(True)
                #     self.sgd_name_list.append(n)
                #     self.sgd_param_list.append(p)
            self.parameters_to_train += list(self.models["encoder"].parameters())    

        else:
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["encoder"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())

            self.models["depth"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["depth"].parameters())



        if self.use_pose_net:
            if self.opt.pose_model_type == 'dino_cls_token':
                self.models['pose'] = networks.depthclip_pose_dinocls.depthclippose_dinocls(768*2)
                self.models["pose"].to(self.device)
                self.parameters_to_train += list(self.models["pose"].parameters())
            elif self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)
                self.models["pose"].to(self.device)
                self.parameters_to_train += list(self.models["pose"].parameters())
                
            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)
                self.models["pose"].to(self.device)
                self.parameters_to_train += list(self.models["pose"].parameters())
                
            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)
                self.models["pose"].to(self.device)
                self.parameters_to_train += list(self.models["pose"].parameters())

            elif self.opt.pose_model_type == "depthclip_pose":
                # self.models["pose"] = networks.depthclippose(self.opt)
                # self.models["pose"].to(self.device)
                
                self.models["pose"] = networks.PoseDecoder(
                self.models["encoder"].num_ch_enc, self.num_pose_frames)
                self.models["pose"].to(self.device)
                
                self.parameters_to_train += list(self.models["pose"].parameters()) 
            elif self.opt.pose_model_type == "depthclip_sep_pose":
                # self.models["pose_encoder"] = networks.depthcliposepencoder(self.opt, pose=True)
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)
                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters()) 
                self.models["pose"] = networks.depthclippose(self.opt)
                self.models["pose"].to(self.device)
                self.parameters_to_train += list(self.models["pose"].parameters())       
                
                for n,p in self.models['pose_encoder'].named_parameters():
                        if 'clip.transformer' in n:
                            p.requires_grad_(False)
                        elif 'logit_scale' in n:
                            p.requires_grad_(False)
                            names.append(n)
                        elif 'clip.text_projection' in n:
                            p.requires_grad_(False)
                            names.append(n)
                        elif 'clip.positional_embedding' in n:
                            p.requires_grad_(False)
                            names.append(n)
                        elif 'clip.token_embedding' in n:
                            p.requires_grad_(False)
                            names.append(n)
                        elif 'clip.ln_final' in n:
                            p.requires_grad_(False)
                            names.append(n)
                        elif 'clip.visual' in n:
                            p.requires_grad_(True)
                            names.append(n)
                        else:
                            p.requires_grad_(True)
                            self.sgd_name_list.append(n)
                            self.sgd_param_list.append(p) 

        # self.opt.config_file = 'vit'

        if self.opt.predictive_mask:  # in orginal monodepthv2，the predictive_mask is False
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())
            
        def get_parameter_number(net):
            total_num = sum(p.numel() for p in net.parameters())
            trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
            return {'Total': total_num, 'Trainable': trainable_num}          
        
        # DOUBLE CHECK
        # for i in self.models.keys():
        #     # for n,p in self.models[str(i)].named_parameters():
        #         # if p.requires_grad==False:
        #         #     print(n)
        #     print(get_parameter_number(self.models[str(i)]))
            
            
        print('================ Trainable Parameter============')
        # for i in self.models.keys():
        #     print("-------------")
            # for n,p in self.models[str(i)].named_parameters():
        for n,p in self.models["encoder"].named_parameters():
            if p.requires_grad==True:
                print(n)            
                                 
                    

        # self.sgd_model_optimizer = optim.SGD(self.sgd_param_list, 0.002, momentum=momentum, weight_decay=weight_decay, dampening=sgd_dampening, nesterov=sgd_nesterov,)
        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate) 
        # self.model_optimizer = optim.AdamW([{'params': self.adam_param_list, "lr": 1e-5}, 
                                        #    {'params': self.sgd_param_list, "lr": 1e-4}])
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        # self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate, betas=(0.5, 0.999))
        # self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(
        #     self.model_optimizer, milestones=[30, 40], gamma=0.5)
        # self.model_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.model_optimizer, mode='min', factor=0.3, patience=4, verbose=True, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        



        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "nyu": datasets.NYURAWDataset,
                         }
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        # self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        self.num_steps_per_epoch = num_train_samples // 1 // self.opt.batch_size
        self.num_total_steps = self.num_steps_per_epoch * self.opt.num_epochs
        
        

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)


        epochs= self.opt.num_epochs
        batch_size = self.opt.batch_size
        total_samples = len(train_dataset)
        total_steps = (total_samples // batch_size) * epochs
        print(total_steps)
        
#         self.model_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.model_optimizer,
#                     max_lr=self.opt.learning_rate,
#                     total_steps=total_steps,
#                     cycle_momentum=True,
#                     base_momentum=0.85, max_momentum=0.95, last_epoch=-1,
#                     div_factor=1,
#                     final_div_factor=100)        
        



        
        # self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate) 
        # self.model_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.model_optimizer, T_max=self.num_total_steps, eta_min=5e-6)

        self.log_file = os.path.join(self.log_path, f'log_{str(time.time()).split(".")[0]}.txt')
        self.writers = {}
        self.writers['train'] = wandb.init(project='cfmde-stage2', name=self.opt.model_name)
        wandb.config.update(self.opt)
        #for mode in ["train", "val"]:
            #self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        # self.adamw_model_lr_scheduler.step()
        # self.sgd_model_lr_scheduler.step()
        
        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            
            before_op_time = time.time() 
            outputs, losses = self.process_batch(inputs)
            # self.adamw_model_optimizer.zero_grad()
            # self.sgd_model_optimizer.zero_grad()
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            # self.adamw_model_optimizer.step()
            # self.sgd_model_optimizer.step()
            duration = time.time() - before_op_time
            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0
            
            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()
           
            self.step += 1
        self.model_lr_scheduler.step()
        print("Current learning rate:", self.model_optimizer.param_groups[0]['lr'])
        
        
        
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])   # 48,3,192,640
            all_features = self.models["encoder"](all_color_aug)                             # 一共有5个，因为encoder是返回多尺度的，第一个为48,64,96,320，第二个为torch.Size([48, 64, 48, 160])，第三个为torch.Size([48, 128, 24, 80])，最后一个为torch.Size([48, 512, 6, 20])
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]     # 将每个分为了frame id的形式，格式为[[16, channel, h, w] , [16, channel, h, w], [16, channel, h, w]]*5 , *5是因为多尺度，*3是因为split batchsize得到的frame id的个数，本来就是根据frame id得到的
            
            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]   # 遍历全部的all_features,取all_feature中每个元素的第0个，即frame id为0的feature
            outputs = self.models["depth"](features[0])
            

        elif self.opt.pose_model_type == "depthclip_pose" :
            
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            outputs, all_features, depth, self.axisangle_dict = self.models["encoder"](all_color_aug, pose=True)
            
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]
            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]
            outputs = self.models["depth"](features[0], depth)  
                
        elif self.opt.pose_model_type == "depthclip_sep_pose" :
            outputs, features, depth, self.axisangle_dict = self.models["encoder"](inputs["color_aug", 0, 0], pose=False)
            outputs = self.models["depth"](features, depth)
            
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            # features = self.models["encoder"](inputs["color_aug", 0, 0])   #只计算一帧的第一个尺度的, 这里的feature其实就是过了个feature map
            # outputs = self.models["depth"](features)
            # outputs, features, _, _ = self.models["depthclip"](inputs["color_aug", 0, 0], pose=False)
            if not self.opt.pose_model_type == 'dino_cls_token':
                outputs, features, depth, self.axisangle_dict = self.models["encoder"](inputs["color_aug", 0, 0], pose=False)
                outputs = self.models["depth"](features, depth)
            else:
                all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
                _, all_features, depth, _ = self.models["encoder"](all_color_aug, pose=False)
                all_features = [torch.split(f, self.opt.batch_size) for f in all_features]
                all_cls_tokens = torch.split(depth['dino_cls_tokens'], self.opt.batch_size)
                
                features = {}
                for i, k in enumerate(self.opt.frame_ids):
                    features[k] = [f[i] for f in all_features]   # 遍历全部的all_features,取all_feature中每个元素的第0个，即frame id为0的feature
                cls_tokens = {}
                for i, k in enumerate(self.opt.frame_ids):
                    cls_tokens[k] = all_cls_tokens[i] 
                outputs = self.models["depth"](features[0], depth)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            if self.opt.pose_model_type == 'dino_cls_token':

                pose_feats = {f_i: cls_tokens[f_i] for f_i in self.opt.frame_ids}

                for f_i in self.opt.frame_ids[1:]:
                    if f_i != "s":
                        # To maintain ordering we always pass frames in temporal order
                        if f_i < 0:
                            pose_inputs = [pose_feats[f_i], pose_feats[0]]
                        else:
                            pose_inputs = [pose_feats[0], pose_feats[f_i]]

                        axisangle, translation = self.models['pose'](torch.cat(pose_inputs, 1))
                                
                        outputs[("axisangle", 0, f_i)] = axisangle
                        outputs[("translation", 0, f_i)] = translation
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
            else:
                outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses


    # def process_batch(self, inputs):
    #     return outputs, losses




    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared"  or  self.opt.pose_model_type == "depthclip_share" or self.opt.pose_model_type == "depthclip_pose":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]  #这是把两帧的concat
                        axisangle, translation = self.models["pose"](pose_inputs)
                        
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)
                        axisangle, translation = self.models["pose"](pose_inputs)

                    elif self.opt.pose_model_type == "depthclip_pose":
                        axisangle, translation = self.models["pose"](pose_inputs, self.axisangle_dict)
                        
                    elif self.opt.pose_model_type == "depthclip_sep_pose":
                        # pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1), pose=True)]  #这是把两帧的concat
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        axisangle, translation, tsne_pose = self.models["pose"](pose_inputs, self.axisangle_dict)
                            
                    else:
                        axisangle, translation = self.models["pose"](pose_inputs)
                    
                    # 12,2,1,3
                    # tsne
                    # from sklearn.manifold import TSNE
                    # import matplotlib.pyplot as plt
                    # tsne_pose = pose_inputs[-1][-1][0]
                    # tsne_depth = features[-1][0]
                    
                    
                    # depth_feature_permuted = tsne_depth.permute(1, 2, 0)  # [6, 20, 2048]

                    # # 应用平均池化将长度从2048降到512
                    # depth_feature_downsampled = F.adaptive_avg_pool1d(depth_feature_permuted, 512)  # [6, 20, 512]

                    # # 调整回原始维度顺序
                    # tsne_depth = depth_feature_downsampled.permute(2, 0, 1).contiguous()  # [512, 6, 20]
                    
                    
                    
                    # pose_feature_flat = tsne_pose.view(-1, 20)
                    # depth_feature_flat = tsne_depth.view(-1, 20)

                    # # 合并特征用于t-SNE
                    # combined_features = torch.cat([pose_feature_flat, depth_feature_flat], dim=0)

                    # # 转换为numpy数组
                    # combined_features_np = combined_features.cpu().detach().numpy()

                    # # 使用t-SNE降维
                    # tsne = TSNE(n_components=2, random_state=42)
                    # combined_features_tsne = tsne.fit_transform(combined_features_np)

                    # # 分离降维后的pose和depth特征
                    # pose_features_tsne = combined_features_tsne[:pose_feature_flat.size(0)]
                    # depth_features_tsne = combined_features_tsne[pose_feature_flat.size(0):]

                    # # 可视化

                    # plt.figure(figsize=(10, 6))
                    # plt.scatter(pose_features_tsne[:, 0], pose_features_tsne[:, 1], label='Pose Features', alpha=0.5)
                    # plt.scatter(depth_features_tsne[:, 0], depth_features_tsne[:, 1], label='Depth Features', alpha=0.5)
                    # plt.legend(fontsize=15)
                    # plt.title('t-SNE of Pose and Depth Features')
                    # # plt.xlabel('t-SNE Dimension 1')
                    # # plt.ylabel('t-SNE Dimension 2')
                    
                    # plt.savefig('tsne_pose_depth_features.png')
                    # # plt.show()
                    # exit(0)
                    
                    
                    
                    
                    
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                    # axisangle[:,0]的维度是：12，1，3
                    
                    
        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            # inputs = self.val_iter.next()
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)  
            outputs[("depth", 0, scale)] = depth
            
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":  

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            # disp = 1/outputs[("depth", 0, scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()
            
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            # loss += 0
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
        with open(self.log_file, 'a') as f:
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)), file=f)

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        current_lr =self.model_optimizer.param_groups[0]['lr']
        #writer = self.writers[mode]
        wandb.log({"learning_rate": current_lr}, step=self.step)
        for l, v in losses.items():
            wandb.log({
                f"{mode}/{l}": v
            }, step=self.step)
            #writer.add_scalar("{}".format(l), v, self.step)
            #writer.add_scalar("learning_rate", current_lr, self.step)
        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    Img = wandb.Image(inputs[("color", frame_id, s)][j].data, caption="color_{}_{}/{}".format(frame_id, s, j))
                    wandb.log({"{}_color_{}_{}/{}".format(mode, frame_id, s, j): Img}, step=self.step)
                    #writer.add_image(
                    #    "color_{}_{}/{}".format(frame_id, s, j),
                    #    inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        Img = wandb.Image(outputs[("color", frame_id, s)][j].data, caption="color_pred_{}_{}/{}".format(frame_id, s, j))
                        wandb.log({"{}_color_pred_{}_{}/{}".format(mode, frame_id, s, j): Img}, step=self.step)
                        #writer.add_image(
                        #    "color_pred_{}_{}/{}".format(frame_id, s, j),
                        #    outputs[("color", frame_id, s)][j].data, self.step)
                
                Img = wandb.Image(normalize_image(outputs[("disp", s)][j]), caption="disp_{}/{}".format(s, j))
                wandb.log({"{}_disp_{}/{}".format(mode, s, j): Img}, step=self.step)
                #writer.add_image(
                #    "disp_{}/{}".format(s, j),
                #    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        Img = wandb.Image(outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...], caption="predictive_mask_{}_{}/{}".format(frame_id, s, j))
                        wandb.log({"{}_predictive_mask_{}_{}/{}".format(mode, frame_id, s, j): Img}, step=self.step)
                        # writer.add_image(
                        #     "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                        #     outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                        #     self.step)

                elif not self.opt.disable_automasking:
                    Img = wandb.Image(outputs["identity_selection/{}".format(s)][j][None, ...], caption="automask_{}/{}".format(s, j))
                    wandb.log({"{}_automask_{}/{}".format(mode, s, j): Img}, step=self.step)
                    #writer.add_image(
                    #    "automask_{}/{}".format(s, j),
                    #    outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)


    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        # if os.path.isfile(optimizer_load_path):
        #     print("Loading Adam weights")
        #     optimizer_dict = torch.load(optimizer_load_path)
        #     self.model_optimizer.load_state_dict(optimizer_dict)
        # else:
        #     print("Cannot find Adam weights so Adam is randomly initialized")
