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
        self.args = OmegaConf.load('/code/CFMDE-main/Stage2/params/basicParams_vit.yaml')
        args = self.args
        # Initialise CLIP
        self.clip, self.clip_preprocess = clip.load(self.args.depthclip.clip, device="cpu", pose=pose, download_root='checkpoints')
        self.use_stereo = False
        self.num_ch_enc = np.array([64, 256, 512, 1024, 2048])

        self._encoder_params_module_list = []
        self._non_encoder_params_module_list = []
        self._frozen_params_module_list = []
        self._zero_params_module_list = []  # Used for CLIP. Requires_grad=True but not passed to optimizer, so shouldn't update...
        
        self._extra_learnable_params_list = []  # Temporary holding place for learnable tokens. If self.args.depthclip.freeze_depthclip is True, these will be frozen. Else they will be learnable.
    

#         self.lenth = 16
#         self.extra_tkns_cfg = {}    # Contains k:v pairs of token name, number of extra tokens with that name.
#         self.extra_tkns_learnables = {}   # Contains refs to extra LEARNABLE tokens, if any.  
#         self.extra_tkns_lookup = {}   # Key = new token, value = learnable parameter for that token.
#         self.extra_tkns_reverse_lookup = {}   # Key = token index, value = learnable parameter for that token.
#         lang_strat = args.depthclip.lang_strat
#         # Templates to use
#         match lang_strat.get("templates"):
#             # Static templates (i.e. the template words are static, but the depth/object tokens can be either.)
#             case "paper":
#                 self.sentence_templates = ["This {object_tkn} is {depth_tkn}"]
#             case "paper-v1":
#                 self.sentence_templates = ["This {object_tkn} appears to be {depth_tkn}."]         
#             case "learned-static-4o12d":
#                 self.sentence_templates = ["<|prompt_0|> <|prompt_1|> <|prompt_2|> <|prompt_3|> {object_tkn} <|prompt_4|> <|prompt_5|> <|prompt_6|> <|prompt_7|> <|prompt_8|> <|prompt_9|> <|prompt_10|> <|prompt_11|> <|prompt_12|> <|prompt_13|> <|prompt_14|> <|prompt_15|> {depth_tkn}."]
#                 self.extra_tkns_cfg["prompt"] = 16
#                 self.extra_tkns_learnables_prompt = nn.ParameterList(values=[torch.normal(mean=0, std=0.02, size=(1, 512)) for i in range(16)])
#                 self._extra_learnable_params_list += self.extra_tkns_learnables_prompt
#                 for i in range(16):
#                     self.extra_tkns_lookup[f"<|prompt_{i}|>"] = self.extra_tkns_learnables_prompt[i]        



#         # Depth tokens to use
#         match lang_strat.get("depth_tokens"):
#             # Static depth tokens

#             case "depth-7":
#                 self.depth_tokens = [
#                     "very close",
#                     "close",
#                     "slightly close",
#                     "neither close nor distant",
#                     "slightly distant",
#                     "distant",
#                     "very distant"
#                 ]

#             # Learned depth tokens
#             case "learned-static-256":
                
#                 # 256 learned depth tokens. NOT input-dependent: represented as learnable parameters.
#                 self.depth_tokens = [f"<|depth_{i}|>" for i in range(self.lenth)]
#                 self.extra_tkns_cfg["depth"] = self.lenth
#                 self.extra_tkns_learnables_depth = nn.ParameterList(values=[torch.normal(mean=0, std=0.02, size=(1, 512)) for i in range(self.lenth)])
#                 self._extra_learnable_params_list += self.extra_tkns_learnables_depth
#                 for i in range(self.lenth):
#                     self.extra_tkns_lookup[f"<|depth_{i}|>"] = self.extra_tkns_learnables_depth[i]
#             case _:
#                 sys.exit("Error: language strategy depth tokens set not recognised")

#         # Object tokens to use
#         match lang_strat.get("object_tokens"):
#             # Static object tokens
#             case "paper":
#                 self.object_tokens = ["object"]
#             case "learned-static-1":
#                 # 1 learned object token. NOT input-dependent: represented as learnable parameters.
#                 self.object_tokens = [f"<|object_{i}|>" for i in range(1)]
#                 self.extra_tkns_cfg["object"] = 1
#                 self.extra_tkns_learnables_object = nn.ParameterList(values=[torch.normal(mean=0, std=0.02, size=(1, 512))])
#                 self._extra_learnable_params_list += self.extra_tkns_learnables_object
#                 for i in range(1):
#                     self.extra_tkns_lookup[f"<|object_{i}|>"] = self.extra_tkns_learnables_object[i]
#             case _:
#                 sys.exit("Error: language strategy object tokens set not recognised")                   




#         # Build templates (word-level). Doesn't handle img_f substitution.
#         self.texts = []
#         for template in self.sentence_templates:
#             for obj in self.object_tokens:
#                 for depth in self.depth_tokens:
#                     self.texts.append(template.format(object_tkn=obj, depth_tkn=depth))
#         self.texts_tokenized, self.tokenizer = clip.tokenize(self.texts, extra_tkns_cfg=self.extra_tkns_cfg)  # tokenize                




        # # ######  axisangle #################
        self.lenth = 16
        axisangle_strat = self.args.depthclip.axisangle_strat
        self.axisangle_extra_tkns_cfg = {}    # Contains k:v pairs of token name, number of extra tokens with that name.
        self.axisangle_extra_tkns_learnables = {}   # Contains refs to extra LEARNABLE tokens, if any.  
        self.axisangle_extra_tkns_lookup = {}   # Key = new token, value = learnable parameter for that token.
        self.axisangle_extra_tkns_reverse_lookup = {}   # Key = token index, value = learnable parameter for that token.
        
        if axisangle_strat.get("axisangle_templates") == 'paper':
            # Static templates (i.e. the template words are static, but the depth/object tokens can be either.)
                self.axisangle_sentence_templates = ["This two {photo_tkn} is {axisangle_tkn_tkn}"]
        elif axisangle_strat.get("axisangle_templates") == "paper-v1":
                self.axisangle_sentence_templates = ["The pose for this two {photo_tkn} appears to be {axisangle_tkn}."]
            
            # Learnable templates. <x>o<y>d means x learnable tokens, then the object token, then y more learnable tokens, then the depth token.
        elif axisangle_strat.get("axisangle_templates") == "learned-static-4o4d":
                # self.axisangle_sentence_templates = ["<|prompt_0|> <|prompt_1|> <|prompt_2|> <|prompt_3|> {photo_tkn} <|prompt_4|> <|prompt_5|> <|prompt_6|> <|prompt_7|> <|prompt_8|> <|prompt_9|> <|prompt_10|> <|prompt_11|> <|prompt_12|> <|prompt_13|> <|prompt_14|> <|prompt_15|> {axisangle_tkn}."]
                self.axisangle_sentence_templates = ["<|prompt_0|> <|prompt_1|> <|prompt_2|> <|prompt_3|> {photo_tkn} <|prompt_4|> <|prompt_5|> <|prompt_6|> <|prompt_7|> {axisangle_tkn}."]
                self.axisangle_extra_tkns_cfg["prompt"] = 16
                self.axisangle_extra_tkns_learnables_prompt = nn.ParameterList(values=[torch.normal(mean=0, std=0.02, size=(1, 512)) for i in range(16)])
                self._extra_learnable_params_list += self.axisangle_extra_tkns_learnables_prompt
                for i in range(16):
                    self.axisangle_extra_tkns_lookup[f"<|prompt_{i}|>"] = self.axisangle_extra_tkns_learnables_prompt[i]
        else:
                sys.exit("Error: sentence templates set not recognised.")        
        
        if axisangle_strat.get("photo_tokens") == 'paper':
            # Static object tokens
                self.axisangle_object_tokens = ["photoes"]
        elif axisangle_strat.get("photo_tokens") == "learned-static-1":
                # 1 learned object token. NOT input-dependent: represented as learnable parameters.
                self.axisangle_object_tokens = [f"<|object_{i}|>" for i in range(1)]
                self.axisangle_extra_tkns_cfg["object"] = 1
                self.axisangle_extra_tkns_learnables_object = nn.ParameterList(values=[torch.normal(mean=0, std=0.02, size=(1, 512))])
                self._extra_learnable_params_list += self.axisangle_extra_tkns_learnables_object
                for i in range(1):
                    self.axisangle_extra_tkns_lookup[f"<|object_{i}|>"] = self.axisangle_extra_tkns_learnables_object[i]
        else:
                sys.exit("Error: language strategy photoes tokens set not recognised")
            
            
        if axisangle_strat.get("axisangle_tokens") == "axisangle":
            # Static depth tokens
                self.axisangle_tokens = [
                    "very small",
                    "small",
                    "slightly small",
                    "neither small nor large",
                    "slightly large",
                    "large",
                    "very large",
                ]

            # Learned depth tokens
        elif axisangle_strat.get("axisangle_tokens") == "learned-static-7":
                # 7 learned depth tokens. NOT input-dependent: represented as learnable parameters.
                self.axisangle_tokens = [f"<|depth_{i}|>" for i in range(7)]
                self.axisangle_extra_tkns_cfg["depth"] = 7
                self.axisangle_extra_tkns_learnables_depth = nn.ParameterList(values=[torch.normal(mean=0, std=0.02, size=(1, 512)) for i in range(7)])
                self._extra_learnable_params_list += self.axisangle_extra_tkns_learnables_depth
                for i in range(7):
                    self.axisangle_extra_tkns_lookup[f"<|depth_{i}|>"] = self.axisangle_extra_tkns_learnables_depth[i]
                    
        elif axisangle_strat.get("axisangle_tokens") == "learned-static-128":
                # 256 learned depth tokens. NOT input-dependent: represented as learnable parameters.
                self.axisangle_tokens = [f"<|depth_{i}|>" for i in range(self.lenth)]
                self.axisangle_extra_tkns_cfg["depth"] = self.lenth
                self.axisangle_extra_tkns_learnables_depth = nn.ParameterList(values=[torch.normal(mean=0, std=0.02, size=(1, 512)) for i in range(self.lenth)])
                self._extra_learnable_params_list += self.axisangle_extra_tkns_learnables_depth
                for i in range(self.lenth):
                    self.axisangle_extra_tkns_lookup[f"<|depth_{i}|>"] = self.axisangle_extra_tkns_learnables_depth[i]

        self.axisangle_texts = []
        for template in self.axisangle_sentence_templates:
            for obj in self.axisangle_object_tokens:
                for axisangle in self.axisangle_tokens:
                    self.axisangle_texts.append(template.format(photo_tkn=obj, axisangle_tkn=axisangle))
        self.axisangle_texts_tokenized, self.axisangle_tokenizer = clip.tokenize(self.axisangle_texts, extra_tkns_cfg=self.axisangle_extra_tkns_cfg)  # tokenize



    def get_axisangle_language_features(self, img_f_tkns=None, extra_tkns_override=None):
        """Get static language (no learned embeddings.
        Uses self.sentence_templates, self.depth_tokens and self.object_tokens to create language embeddings
        for each depth class.
        Follows function zeroshot_classifier() in DepthCLIP/monoclip.py, from original DepthCLIP paper's code.
        
        if img_f_tkns is not None, then it's a grid of tokens from img_f, dims BxHWx512. 512 is to match the token size of CLIP's tokens.
        They're assumed to be object tokens (i.e. to replace object_tkns)
        """

        self.axisangle_texts_tokenized = self.axisangle_texts_tokenized.cuda()

        # If there are extra tokens to be used, and if the index-to-param lookup hasn't been initialised, then do that
        # Doing this here is necessary because self.device isn't set properly during __init__().
        # Overwrite img_f token if needed
        if self.axisangle_extra_tkns_lookup != {} and (self.axisangle_extra_tkns_reverse_lookup == {}):
            for k, v in self.axisangle_tokenizer.extra_tkns_reverse_lookup.items():
                word_tkn = self.axisangle_tokenizer.extra_tkns_reverse_lookup[k]
                if "imgf" in word_tkn:
                    self.axisangle_extra_tkns_reverse_lookup[k] = word_tkn, None
                else:
                    self.axisangle_extra_tkns_reverse_lookup[k] = (word_tkn, self.axisangle_extra_tkns_lookup[word_tkn].cuda())

        if extra_tkns_override is not None:
            for i, (k, v) in enumerate(self.axisangle_extra_tkns_reverse_lookup.items()):
                self.axisangle_extra_tkns_reverse_lookup[k] = extra_tkns_override[:, i]   # Each is Bx512 now. encode_text should handle this.

        text_embeddings = self.clip.encode_text(self.axisangle_texts_tokenized, self.axisangle_extra_tkns_reverse_lookup, self.axisangle_tokenizer.extra_tkns_fwd_lookup, img_f_tkns)  # embed with text encoder

        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        text_embedding = text_embeddings / text_embeddings.norm()

        return text_embedding.swapaxes(-2, -1).contiguous()    # permute is done because the old way used to stack on dim=1, needs to return C, N
    
    
        
    def txt_to_language_features(self, sentences):
        """ Runs a list of sentences through CLIP as a single batch. Does not account for extra tokens.
        
        Args:
            :param sentences (list of strings): The batch of N sentences to embed
        
        Returns:
            :returns txtf (torch.Tensor, NxC): The embedded text features.
        """
        tokenized, tokenizer = clip.tokenize(sentences)  # tokenize
        tokenized = tokenized.cuda()

        text_embeddings = self.clip.encode_text(tokenized)  # embed with text encoder

        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        text_embedding = text_embeddings / text_embeddings.norm()

        return text_embedding.swapaxes(-2, -1).contiguous()    # permute is done because the old way used to stack on dim=1, needs to return C, N

    def get_language_features(self, img_f_tkns=None, extra_tkns_override=None):
        """Get static language (no learned embeddings.
        Uses self.sentence_templates, self.depth_tokens and self.object_tokens to create language embeddings
        for each depth class.
        Follows function zeroshot_classifier() in DepthCLIP/monoclip.py, from original DepthCLIP paper's code.
        
        if img_f_tkns is not None, then it's a grid of tokens from img_f, dims BxHWx512. 512 is to match the token size of CLIP's tokens.
        They're assumed to be object tokens (i.e. to replace object_tkns)
        """

        self.texts_tokenized = self.texts_tokenized.cuda()

        # If there are extra tokens to be used, and if the index-to-param lookup hasn't been initialised, then do that
        # Doing this here is necessary because self.device isn't set properly during __init__().
        # Overwrite img_f token if needed
        if self.extra_tkns_lookup != {} and (self.extra_tkns_reverse_lookup == {}):
            for k, v in self.tokenizer.extra_tkns_reverse_lookup.items():
                word_tkn = self.tokenizer.extra_tkns_reverse_lookup[k]
                if "imgf" in word_tkn:
                    self.extra_tkns_reverse_lookup[k] = word_tkn, None
                else:
                    self.extra_tkns_reverse_lookup[k] = (word_tkn, self.extra_tkns_lookup[word_tkn].cuda())

        if extra_tkns_override is not None:
            for i, (k, v) in enumerate(self.extra_tkns_reverse_lookup.items()):
                self.extra_tkns_reverse_lookup[k] = extra_tkns_override[:, i]   # Each is Bx512 now. encode_text should handle this.

        text_embeddings = self.clip.encode_text(self.texts_tokenized, self.extra_tkns_reverse_lookup, self.tokenizer.extra_tkns_fwd_lookup, img_f_tkns)  # embed with text encoder

        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        text_embedding = text_embeddings / text_embeddings.norm()

        return text_embedding.swapaxes(-2, -1).contiguous()    # permute is done because the old way used to stack on dim=1, needs to return C, N


    def forward(self, image, pose):
        
        # pose
        img_f_tkns = None
        extra_tkns_override = None 
        axisangle_txt_f = self.get_axisangle_language_features(img_f_tkns, extra_tkns_override)
        axisangle = OrderedDict()
        axisangle['axisangle_txt_f'] = axisangle_txt_f
        
        
        # depth decoder
        img_f_tkns = None
        extra_tkns_override = None
#         txt_f = self.get_language_features(img_f_tkns, extra_tkns_override)      # 3 dims
        depth = OrderedDict()
#         depth['tex_f'] = txt_f
        
        
        # encoder
        img_f, features, _, _ = self.clip.encode_image(image, pose)

        return img_f, features, depth, axisangle




