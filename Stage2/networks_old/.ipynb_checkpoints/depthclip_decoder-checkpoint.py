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
from .CBAM import CBAM
class depthclipdecoder(nn.Module):
    """
    A semi-faithful implementation of DepthCLIP, with various modifications.
    Based heavily on DepthCLIP from "Can Language Understand Depth?", Zhang et al., 2022, in ACM Multimedia 2022.
    The general idea is to align depth-related language and image features using a pretrained CLIP model.
    This implementation does not exactly follow that in the paper, and also contains various modifications and experiments.
    """
    def __init__(self, args):
        super().__init__()
        
        self.scales = args.scales
        self.args = OmegaConf.load('/code/depth_estimation/monodepth2/basicParams_vit.yaml')
        args = self.args


        # self.conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        # nn.init.zeros_(self.conv_layer.weight)
        # nn.init.zeros_(self.conv_layer.bias)
        # nn.init.normal_(self.conv_layer.weight)
        # nn.init.normal_(self.conv_layer.bias)
        self.relu = nn.ReLU()

        # Decoder
        self.num_output_channels = 1
        self.use_skips = True 
        self.upsample_mode = 'nearest'
        # self.scales = [0]
#         self.num_ch_enc = np.array([self.lenth, self.lenth, self.lenth, self.lenth, self.lenth])
        self.num_ch_enc = np.array([64, 256, 512, 1024, 2048])
    
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        
        self.convs = OrderedDict()
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
#             self.convs[("cbam",i,0)] = CBAM(num_ch_in,16)
            # self.convs[("dem", i)] = DEM(num_ch_in)

        for s in self.scales:
            # self.convs[("dem", i)] = DEM(self.num_ch_dec[s])
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels).cuda()
        # self.convs[("dispconv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_channels).cuda()

        # self.up_conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1))
        # nn.init.constant_(self.up_conv_layer.weight, 1.0)
        # nn.init.constant_(self.up_conv_layer.bias, 0.0)
        # self.spm = SPM(self.num_ch_enc[-1])
        
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()


        

        




    def forward(self, input_features, depth):


        self.outputs = {}

        # decoder
        x = input_features[-1]
        # x = self.spm(x)
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            # x = self.convs[("dem", i)](x)
#             x = self.convs[("cbam", i, 0)](x)
            x = self.convs[("upconv", i, 1)](x)
            
            if i in self.scales:
                # x = self.convs[("dem", i)](x)
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
        return self.outputs
