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

from .dpt import DPTHead
class depthclipdecoder(nn.Module):
    """
    A semi-faithful implementation of DepthCLIP, with various modifications.
    Based heavily on DepthCLIP from "Can Language Understand Depth?", Zhang et al., 2022, in ACM Multimedia 2022.
    The general idea is to align depth-related language and image features using a pretrained CLIP model.
    This implementation does not exactly follow that in the paper, and also contains various modifications and experiments.
    """
    def __init__(self,                
                 args, 
                 in_channels=768,
                 features=256, 
                 out_channels=[256, 512, 1024], 
                 use_bn=False, 
                 use_clstoken=False, 
                 localhub=True,
                 n_depth_tokens=256):
        super().__init__()
        
        # self.scales = args.scales
        self.args = args
        # dim = 768 
        # the decoder
        self.depth_head = DPTHead(1, in_channels, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        self.lenth = n_depth_tokens

    def forward(self, input_features, depth):
        self.outputs = {}
        # h = 192 
        # w = 640
        # patch_h = 192//16
        # patch_w = 640 // 16
        h = 224
        w = 672
        patch_h = h // 14
        patch_w = w // 14

        if self.args.use_depth_text_align:
            depth_text_embeddings = depth['depth_text_embeddings']
            text_aligned_img_features = []    
            for input_feature in input_features:
                hf, wf = input_feature.shape[2], input_feature.shape[3]
                f = input_feature.reshape(input_feature.shape[0], input_feature.shape[1], -1).permute(0, 2, 1)
                f = f/f.norm(dim=-1, keepdim=True)
                img_f = F.interpolate(f, size=depth_text_embeddings.shape[0])
                depth_logits = 100.0 * (img_f @ depth_text_embeddings)
                depth_logits = depth_logits.permute(0, 2, 1).reshape(f.shape[0], self.lenth, hf, wf).contiguous()
                if self.args.cat_depth_text_logic:
                    text_aligned_img_features.append(torch.cat([depth_logits, input_feature], dim=1))
                else:
                    text_aligned_img_features.append(depth_logits)
                
        else:
            text_aligned_img_features = input_features

        depth = self.depth_head(text_aligned_img_features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        self.outputs[("disp", 0)] = depth
        return self.outputs
