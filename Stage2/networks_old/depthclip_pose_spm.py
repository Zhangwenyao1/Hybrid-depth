# DepthCLIP.py
# Based heavily on DepthCLIP from "Can Language Understand Depth?", Zhang et al., 2022, in ACM Multimedia 2022
# The general idea is to align depth-related language and image features using a pretrained CLIP model.
# This implementation does not exactly follow that in the paper, and also contains various modifications and experiments.

from collections import namedtuple, OrderedDict
import math
import os, sys
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import re
from layers import *
from CLIP import clip
import pytorch_lightning as pl
import sys
from modules.LearnableTokenEmbeddings import LearnableTokenEmbeddings
from omegaconf import OmegaConf
import numpy as np
from .spm import SPM


class depthclippose(nn.Module):
    """
    A semi-faithful implementation of DepthCLIP, with various modifications.
    Based heavily on DepthCLIP from "Can Language Understand Depth?", Zhang et al., 2022, in ACM Multimedia 2022.
    The general idea is to align depth-related language and image features using a pretrained CLIP model.
    This implementation does not exactly follow that in the paper, and also contains various modifications and experiments.
    """
    def __init__(self, args):
        super().__init__()
        # self.args = args.config_file
        self.scales = args.scales
        self.args = OmegaConf.load('/code/depth_estimation/monodepth2/basicParams_vit.yaml')
        args = self.args

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # self.fcaxisangle = nn.Linear(120, 3, bias=True)
        # self.fctranslation = nn.Linear(120, 3, bias=True)
        # self.conv_axisangle = nn.Conv2d(4096, 1024, 1)
        # self.bn = nn.BatchNorm2d(1024)
        # self.conv_axisangle_1 = nn.Conv2d(4096, 2048, 1)
        # self.bn1 = nn.BatchNorm2d(2048)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv_axisangle_2 = nn.Conv2d(2048, 1024, 1)
        # self.bn2 = nn.BatchNorm2d(1024)
        # self.relu2 = nn.ReLU(inplace=True)
        # nn.init.zeros_(self.fcaxisangle.weight)
        # nn.init.zeros_(self.fcaxisangle.bias)
        # nn.init.zeros_(self.fctranslation.weight)
        # nn.init.zeros_(self.fcaxisangle.bias)
        # nn.init.zeros_(self.conv_axisangle.weight)
        # nn.init.zeros_(self.conv_translation.weight)
        # self.axisangle_linear = nn.Linear(2048, 1024, bias=True)
        # self.translation_linear = nn.Linear(2048, 1024, bias=True)        
        
        

        
        
        
 

        # Posedecoder
        

        self.lenth = 16
        self.num_frames_to_predict_for = 1

        self.pose_convs = OrderedDict()
        self.pose_convs[('spm')] = SPM([1024])
        # self.pose_convs[("squeeze")] = nn.Conv2d(2048, 512, 1)
        self.pose_convs[('squeeze')] = nn.Conv2d(4096, 1024, 1)
        self.pose_convs[("pose", 0)] = nn.Conv2d(self.lenth, 256, 3, 1, 1)
        self.pose_convs[("pose", 1)] = nn.Conv2d(256, 256, 3, 1, 1)
        self.pose_convs[("pose", 2)] = nn.Conv2d(256, 6 * self.num_frames_to_predict_for, 1)
                                                                                
        self.axisangle_net = nn.ModuleList(list(self.pose_convs.values()))




    def forward(self, input_features, axisangle_dict):

        last_features = [f[-1] for f in input_features]
        cat_features = [f for f in last_features]
        axisangle_txt_f = axisangle_dict['axisangle_txt_f'] 
        
        # cat_features = [self.relu(self.pose_convs["squeeze"](f)) for f in last_features]
        axisangle_cat_features = torch.cat(cat_features, 1) # b, 1024, h
        axisangle = self.relu(self.pose_convs[('squeeze')](axisangle_cat_features))
        # axisangle_cat_features = torch.cat(cat_features, dim=1)
        
        axisangle_feature = axisangle
        # axisangle = self.relu(self.conv_axisangle(axisangle_cat_features))
        # axisangle = self.relu(self.bn1(self.conv_axisangle_1(axisangle_cat_features)))
        # axisangle = self.relu(self.bn2(self.conv_axisangle_2(axisangle)))
        
        axisangle = axisangle.reshape(axisangle.shape[0],axisangle.shape[1],-1).permute(0,2,1)
        axisangle_logits = 100.0 * (axisangle @ axisangle_txt_f) 
        axisangle_logits = axisangle_logits.permute(0, 2, 1).reshape(axisangle_logits.shape[0], axisangle_logits.shape[2], axisangle_cat_features.shape[2], axisangle_cat_features.shape[3])
        # x = axisangle_logits
        
        out = self.pose_convs[('spm')](axisangle_feature, axisangle_logits, 0)
        # axisangle_logits = axisangle_logits.permute(0, 2, 1).reshape(axisangle_logits.shape[0], axisangle_logits.shape[2], -1)
        # axisangle_logits /= temperature
        # axisangle_pred = F.softmax(axisangle_logits, dim=1)
        # axisangle_bin_tensor = torch.tensor(axisangle_bin_centres).to(device)
        # axisangle_pred = axisangle_pred * axisangle_bin_tensor.reshape(1, self.lenth).unsqueeze(-1)
        # axisangle_pred = axisangle_pred.sum(1, keepdim=True)
        # axisangle_pred = 0.001 * self.fcaxisangle(axisangle_pred).unsqueeze(1)
        # for i in range(3):
        #     axisangle_pred = self.axisangle_convs[("pose", i)](axisangle_pred)
        #     if i != 2:
        #         axisangle_pred = relu(axisangle_pred)
                
        # axisangle_pred = axisangle_pred.mean(3).mean(2)

        # axisangle = 0.01 * axisangle_pred.view(-1, 1, 1, 3)
 
        # translation = relu(self.conv_translation(translation_cat_features))
        # translation = translation.reshape(translation.shape[0],translation.shape[1],-1).permute(0,2,1)
        # translation_logits = 100.0 * (translation @ translation_txt_f) 
        # translation_logits = translation_logits.permute(0, 2, 1).reshape(translation_logits.shape[0], translation_logits.shape[2], -1)
        # translation_logits /= temperature
        # translation_pred = F.softmax(translation_logits, dim=1)
        # translation_bin_tensor = torch.tensor(translation_bin_centres).to(device)
        # translation_pred = translation_pred * translation_bin_tensor.reshape(1, self.lenth).unsqueeze(-1)
        # translation_pred = translation_pred.sum(1, keepdim=True)
        # translation_pred = 0.001 * self.fctranslation(translation_pred).unsqueeze(1)
        
        # for i in range(3):
        #     translation_pred = self.translation_convs[("pose", i)](translation_pred)
        #     if i != 2:
        #         translation_pred = relu(translation_pred)
        
        # translation_pred = translation_pred.mean(3).mean(2)

        # translation = 0.01 * translation_pred.view(-1, 1, 1, 3)
 
 
        # out_features = [f for f in last_features]
        # out_cat_features = torch.cat(out_features, dim=1)
        
        # out = out_cat_features
        # out = x
        for i in range(3):
            out = self.pose_convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
        
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, 1, 1, 6)

        # axisangle = 0.5 * out[..., :3] + 0.5 * axisangle_pred
        # translation = 0.5 * out[..., 3:] + 0.5 * translation_pred    
        
        axisangle =out[..., :3]
        translation =out[..., 3:]
        
        
        
        
        # 



        return axisangle, translation
        # return outputs, features, axisangle, translation
        # return outputs, axisangle, translation