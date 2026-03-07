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
from CLIP import clip
from modules.LearnableTokenEmbeddings import LearnableTokenEmbeddings
from omegaconf import OmegaConf
import numpy as np
from  networks import depthclip_encoder 
from .spm import SPM
from .dem import DEM

class depthclipdecoder(nn.Module):
    """
    A semi-faithful implementation of DepthCLIP, with various modifications.
    Based heavily on DepthCLIP from "Can Language Understand Depth?", Zhang et al., 2022, in ACM Multimedia 2022.
    The general idea is to align depth-related language and image features using a pretrained CLIP model.
    This implementation does not exactly follow that in the paper, and also contains various modifications and experiments.
    """
    def __init__(self, args):
        super().__init__()
        
        # self.scales = args.scales
        self.args = OmegaConf.load('/code/depth_estimation/monodepth2/basicParams_vit.yaml')
        args = self.args


        self.temperature = 0.1

        self.relu = nn.ReLU()

        
        # Decoder
        self.num_output_channels = 1
        self.use_skips = True 
        self.upsample_mode = 'bilinear'
        self.scales = [0]
        self.lenth = 16
        self.num_ch_enc = np.array([self.lenth, self.lenth, self.lenth, self.lenth, self.lenth])
        # self.num_ch_dec = np.array([16, 64, 256, 512, 1024])
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.num_ch_original = np.array([64, 256, 512, 1024, 2048])
        
        self.convs = OrderedDict()
        self.convs[('spm',0)] = SPM(self.num_ch_original)
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out).cuda()

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out).cuda()
            # self.convs[("dem", i)] = DEM(num_ch_in)
            
        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels).cuda()
            
        last_inp_channels = np.sum(self.num_ch_original)
        ocr_mid_channels = 32
        self.convs[("aux_head")] = nn.Sequential(
            nn.Conv2d(last_inp_channels, 256,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 16,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.convs[("conv3x3_ocr")] = nn.Sequential(
            nn.Conv2d(last_inp_channels, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True),
        )



        # self.convs[("dispconv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_channels).cuda()
        # self.convs[("dem", i)]
        self.spm = SPM(16)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()



    def forward(self, input_features, depth):

        txt_f = depth['tex_f'] 
        
        align_feature = []
        for f in input_features:
            h, w = f.shape[2], f.shape[3]
            f = f.reshape(f.shape[0], f.shape[1], -1).permute(0, 2, 1)
            f = f/f.norm(dim=-1, keepdim=True)
            img_f = F.interpolate(f, size=txt_f.shape[0])
            depth_logits = 100.0 * (img_f @ txt_f)
            depth_logits = depth_logits.permute(0, 2, 1).reshape(f.shape[0], self.lenth, h, w).contiguous()
            align_feature.append(depth_logits)
        # combine_feature = []
        # for i in range(5):
        #     combine_feature.append(self.convs[('spm',0)](input_features[i], align_feature[i], i))
        # x = input_features
        # x0_h, x0_w = input_features[0].size(2), input_features[0].size(3)
        
        # x0 = F.interpolate(x[0], size=(x0_h, x0_w),
        #                 mode='bilinear', align_corners=True)
        # x1 = F.interpolate(x[1], size=(x0_h, x0_w),
        #                 mode='bilinear', align_corners=True)
        # x2 = F.interpolate(x[2], size=(x0_h, x0_w),
        #                 mode='bilinear', align_corners=True)
        # x3 = F.interpolate(x[3], size=(x0_h, x0_w),
        #                 mode='bilinear', align_corners=True)
        # x4 = F.interpolate(x[4], size=(x0_h, x0_w),
        #                 mode='bilinear', align_corners=True)
        
        # feats = torch.cat([x0, x1, x2, x3, x4], 1)
        
        # align_feature = self.convs[("aux_head")](feats)
        # feats = self.convs[("conv3x3_ocr")](feats)
        
        # combine_feature = []
        # for i in range(5):
        #     combine_feature.append(self.convs[('spm',0)](input_features[i], align_feature[i], i))
        # combine_feature =  self.convs[('spm',0)](feats, align_feature, 0)   
        # outputs = {}
        # combine_feature = F.interpolate(combine_feature, size=(x0_h*2, x0_w*2),
                        # mode='bilinear', align_corners=True)
        # outputs[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](combine_feature))
        
        outputs = {}
        x = align_feature[-1]
        # x = combine_feature[-1]
        # # x = self.spm(input_features[-1], x)
        
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                # if x[0].shape[-1]==align_feature[i - 1].shape[-1]:
                #     x += [align_feature[i - 1]]
                # else: 
                #     x += [upsample(align_feature[i - 1])]
                x += [align_feature[i - 1]]
                # x += [combine_feature[i - 1]] 
            x = torch.cat(x, 1)
            # x = self.convs[("dem", i)](x)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                # outputs[("disp", i)] = (0.5*self.sigmoid(self.convs[("dispconv", i)](x))+0.5*self.sigmoid(self.scalor/depth_pred_inter))
                # outputs[("disp", i)] = self.up_conv_layer(0.5*self.sigmoid(self.convs[("dispconv", i)](x))+0.5*(self.scalor/depth_pred_inter))
                # outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                # outputs[("disp", i)]  = self.sigmoid(self.up_conv_layer(0.5*self.convs[("dispconv", i)](x)+ 0.5*(self.scalor/depth_pred_inter)))
                # outputs[("disp", i)] = self.sigmoid(self.up_conv_layer(self.scalor/depth_pred_inter))
        return outputs
