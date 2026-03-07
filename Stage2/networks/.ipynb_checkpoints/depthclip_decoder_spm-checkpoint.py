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


        self.temperature = 0.1
        # self.conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        # nn.init.zeros_(self.conv_layer.weight)
        # nn.init.zeros_(self.conv_layer.bias)
        # nn.init.normal_(self.conv_layer.weight)
        # nn.init.normal_(self.conv_layer.bias)
        self.relu = nn.ReLU()
        # self.scalor = torch.tensor(1.0, requires_grad=True)
        # self.scalor = nn.Parameter(self.scalor)
        # if "RN" in self.args.depthclip.clip:
        # # rn50
        #     self.transposed_conv_layer = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=32, stride=32, padding=0)

        # #vit b16
        # elif "V" in self.args.depthclip.clip:
        #     self.transposed_conv_layer = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=16, stride=16, padding=0)

        # nn.init.zeros_(self.transposed_conv_layer.weight)
        # nn.init.zeros_(self.transposed_conv_layer.bias)
        # nn.init.normal_(self.transposed_conv_layer.weight,mean=0,std=1)
        # nn.init.normal_(self.transposed_conv_layer.bias,mean=0,std=1)

  
        # Default for extra tokens to add to the tokenizer vocabulary
        # Will get overridden whenever there are learnable tokens to be added. Each token is replaced with its learned counterpart
        # as part of the tokenization process (when calling clip.tokenize(text, extra_tkns)).

        
        
        # Decoder
        self.num_output_channels = 1
        self.use_skips = True 
        self.upsample_mode = 'bilinear'
        # self.scales = [0]
        self.lenth = 16
        self.num_ch_enc = np.array([self.lenth, self.lenth, self.lenth, self.lenth, self.lenth])
        # self.num_ch_dec = np.array([16, 64, 256, 512, 1024])
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


        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels).cuda()

        # self.convs[("dispconv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_channels).cuda()

        # self.up_conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1))
        # nn.init.constant_(self.up_conv_layer.weight, 1.0)
        # nn.init.constant_(self.up_conv_layer.bias, 0.0)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()



        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        # Posedecoder
        
        self.num_input_features = 2

        
        # self.num_frames_to_predict_for = 1

        # self.pose_convs = OrderedDict()
        # self.pose_convs[("squeeze")] = nn.Conv2d(16, 256, 1)
        # self.pose_convs[("posef.net = nn.ModuleList(list(self.pose_convs.values()))

        




    def forward(self, input_features, depth):


        # if "ViT" in self.args.depthclip.clip:
        #     out_height = int(self.in_height / self.clip.visual.patch_size)
        #     out_width = int(self.in_width / self.clip.visual.patch_size)        
        # elif "RN50" in self.args.depthclip.clip:
        #     out_height = int(self.in_height / 32)
        #     out_width = int(self.in_width / 32)
        # else:
        #     pass

        # Run through most of the CLIP encoder to get dense CLIP features
        ## For RN50 backbone, C = 2048
        ## For ViT-B/32 backbone, C = 512
        # self.depth_bin_centres = depth['depth_bin_centres']
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

        outputs = {}
        x = align_feature[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                if x[0].shape[-1]==align_feature[i - 1].shape[-1]:
                    x += [align_feature[i - 1]]
                else: 
                    x += [upsample(align_feature[i - 1])]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                # outputs[("disp", i)] = (0.5*self.sigmoid(self.convs[("dispconv", i)](x))+0.5*self.sigmoid(self.scalor/depth_pred_inter))
                # outputs[("disp", i)] = self.up_conv_layer(0.5*self.sigmoid(self.convs[("dispconv", i)](x))+0.5*(self.scalor/depth_pred_inter))
                # outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                # outputs[("disp", i)]  = self.sigmoid(self.up_conv_layer(0.5*self.convs[("dispconv", i)](x)+ 0.5*(self.scalor/depth_pred_inter)))
                # outputs[("disp", i)] = self.sigmoid(self.up_conv_layer(self.scalor/depth_pred_inter))
        # for i in self.scales:
        #     outputs[("disp", i)] = 1/depth_pred # absolute depth     depth->disparity  depth formula  set: scalor/depth = disparity 
                
        # outputs[("disp", 0)] = self.sigmoid(self.up_conv_layer(self.scalor/depth_pred_inter))
        return outputs
