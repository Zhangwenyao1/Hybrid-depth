# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in
file_dir = '/data/zhangwenyao/drive_data'

class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")
        self.parser.add_argument("--vis_dir", type=str, default=None)
        self.parser.add_argument("--only_dino", action='store_true', default=False)
        self.parser.add_argument("--use_depth_text_align", action='store_true', default=False)
        self.parser.add_argument("--n_depth_text_tokens", type=int, default=1024)
        self.parser.add_argument("--stage1_checkpoint_path", type=str, default="")
        self.parser.add_argument("--dino_clip_align_method", type=str, default="interp_conv1x1")
        self.parser.add_argument("--use_depth_text_embedding_preinit", action='store_true', default=False)
        self.parser.add_argument("--cat_depth_text_logic", action="store_true", default=False)
        self.parser.add_argument("--only_clip", action="store_true", default=False)
        self.parser.add_argument("--use_resnet50_instead_clip", action="store_true", default=False)
        self.parser.add_argument("--resnet50_random_init", action="store_true", default=False)

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 #default="/data/zhangwenyao/drive_data/kitti_dataset_copy/raw")
                                 default='/data/ericliu/KITTI-depth/kitti_dataset_copy/raw')
                                 
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default='/code/CFMDE-main/Stage2/exps')
                                 #default=os.path.join(os.path.expanduser("~"), "tmp"))
        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark", "nyu",],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                              #    default = "nyu", 
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", "nyu"])
        self.parser.add_argument("--png",
                                 default=False,
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
#                                  default=224)
#                                 default=192)
                                 default=224)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
#                                  default=224
#                                 default=640,
                                 default=672,
                                 )
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 # default=[0, 1, 2, 3])
                               default=[0])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        
      #   self.parser.add_argument("--config_file", 
      #                            type=argparse.FileType('r', encoding='UTF-8'), 
      #                            help="Path to the config/params YAML file.",
      #                            default='/code/depth_estimation/monodepth2/basicParams_vit.yaml')


        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=16)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
                                 # default=0.00035)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=30)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)
        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 # default=True,
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 # default="depthclip_pose_text_two",
                                 # default= "depthclip_sep_pose",
                                 default = 'separate_resnet',
                                 #default = 'dino_cls_token',
                                 choices=["posecnn", "separate_resnet", "shared", "depthclip_share","depthclip_pose","depthclip_sep_pose", "dino_cls_token"])
        self.parser.add_argument("--depth_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="depthclip_ed",
                                 # default = 'separate_resnet',
                                 choices=["depth", "depthclip, depthclip_ed"])
        
         # lora
         
        self.parser.add_argument('--backbone', default='ViT-B/16', type=str)
        self.parser.add_argument('--position', 
                                 type=str, 
                                 default='all', 
                                 choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], 
                                 help='where to put the LoRA modules')
        self.parser.add_argument('--encoder', 
                                 type=str, 
                                 choices=['text', 'vision', 'both'], 
                                 default='both')
        self.parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
        self.parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
        self.parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
        self.parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')
        
        
        
        
        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=8)

        # LOADING options
        self.parser.add_argument("--eval_all", action='store_true', default=False)
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 # default="/code/depth_estimation/monodepth2/results/0320_sep_monoposewithtext/models/weights_29",
                              #    default = "/code/occ/PromptLearningCLIP-MDE/runs/basicParams/version_144/",
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 # default="eigen",
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10","nyu","eigen_improved"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 default=False,
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
