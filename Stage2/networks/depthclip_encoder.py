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

import pytorch_lightning as pl
from CLIP import clip
import sys
from modules.LearnableTokenEmbeddings import LearnableTokenEmbeddings
from omegaconf import OmegaConf
import numpy as np
# from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
# from loralib.run_utils import *
class depthclipencoder(nn.Module):
    """
    A semi-faithful implementation of DepthCLIP, with various modifications.
    Based heavily on DepthCLIP from "Can Language Understand Depth?", Zhang et al., 2022, in ACM Multimedia 2022.
    The general idea is to align depth-related language and image features using a pretrained CLIP model.
    This implementation does not exactly follow that in the paper, and also contains various modifications and experiments.
    """
    def __init__(self, args, pose=False):
        super().__init__()
        # self.args = args.config_file
        # self.scales = args.scales
        # self.args = OmegaConf.load('/code/depth_estimation/monodepth2/basicParams_vit.yaml')
        # Initialise CLIP
        self.args = args
        self.clip, self.clip_preprocess = clip.load(self.args.backbone, device="cpu", pose=pose, download_root='/code/CFMDE-main/checkpoints')
        #apply_lora(args, self.clip)

    def forward(self, image, pose):
       # encoder
        # image = self.clip_preprocess(image)
        img_f, features = self.clip.encode_image(image)
        depth = None
        axisangle = None
        return img_f, features, depth, axisangle




