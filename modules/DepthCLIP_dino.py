# DepthCLIP.py
# Based heavily on DepthCLIP from "Can Language Understand Depth?", Zhang et al., 2022, in ACM Multimedia 2022
# The general idea is to align depth-related language and image features using a pretrained CLIP model.
# This implementation does not exactly follow that in the paper, and also contains various modifications and experiments.

from collections import namedtuple
import math
import os, sys
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import re

import pytorch_lightning as pl
from CLIP import clip   # Ensure that CLIP is being imported from the submodule, to make use of local modifications.
from modules.LearnableTokenEmbeddings import LearnableTokenEmbeddings
from modules.dpt import DPTHead

class ImgFInterpolateWrapper():
    """ A wrapper for nn.functional.interpolate() """
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    
    def __call__(self, image, img_f, img_f_hw):
        """
        image: the image input to the whole model. Used for size reference 
        img_f: The img_f to be resized.
        img_f_hw: tuple of (H, W) of the image features.
        """
        img_f = img_f.reshape(img_f.shape[0], img_f_hw[0], img_f_hw[1], img_f.shape[-1]).permute(0, 3, 1, 2)

        out_h = int(image.shape[2] * self.scale_factor)
        out_w = int(image.shape[3] * self.scale_factor)
        img_f = F.interpolate(img_f, size=[out_h, out_w], mode="bilinear", align_corners=False)
        img_f = img_f.reshape(img_f.shape[0], img_f.shape[1], -1).permute(0, 2, 1).contiguous()

        return img_f, (out_h, out_w)


class BasicDecoder(nn.Module):
    """ A simple decoder module consisting of nearest neighbour upsample + 3x3 conv + ReLU blocks.
    It will have num_decoder_blocks blocks."""
    def __init__(self, num_decoder_blocks, in_channels, channel_reduce = True):
        """
        :param num_decoder_blocks (int): how many sets of upsample + conv to do to the input
        :param in_channels (int): Number of channels expected at the input
        :param channel_reduce (bool): whether or not to halve the number of channels each time. If false, won't reduce channels.
        """
        super().__init__()
        self.num_decoder_blocks = num_decoder_blocks
        self.in_channels = in_channels
        self.decoder_block_list = []
        curr_in_channels = self.in_channels
        for i in range(self.num_decoder_blocks):
            curr_out_channels = curr_in_channels // (2 ** i) if channel_reduce else curr_in_channels
            self.decoder_block_list.append(nn.Conv2d(curr_in_channels, curr_out_channels, kernel_size=3, stride=1, padding=1))
            curr_in_channels = curr_out_channels
        
        self.decoder_block_list = nn.ModuleList(self.decoder_block_list)

    
    def forward(self, input):
        x = input
        for i in range(len(self.decoder_block_list)):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = F.relu(self.decoder_block_list[i](x))
        
        return x


class UpSampleWithSkip(nn.Module):
    """Upsamples feature input to dimensions of skip connection, concatenates the two in the channel dimension,
    then runs through two conv/bn/leakyRelu blocks.
    This mirrors the implementation of the layers used in the AdaBins decoder.
    """
    def __init__(self, input_features, output_features):
        super().__init__()

        self._net = nn.Sequential(nn.Conv2d(input_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, skip_features):
        up_x = F.interpolate(x, size=[skip_features.size(2), skip_features.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, skip_features], dim=1)
        return self._net(f)


class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Img_f_agg2tkns(nn.Module):
    """ Take img_f_agg (Bx1xC), concat to tokens, then run through MLP to get final image-dependent tokens."""

    def __init__(self, img_f_agg_dims=1024, tkn_dims=512):
        super().__init__()
        self.img_f_agg2tkndim = MLP((img_f_agg_dims + tkn_dims, tkn_dims, tkn_dims), act=nn.Tanh)   # Convert to 512 dimension


    def forward(self, img_f_agg: torch.Tensor, pre_tkns: List[torch.Tensor]) -> torch.Tensor:
        """ Params:
        :param img_f_agg (torch.Tensor, Bx1xC): Image features for whole image from CLIP visual encoder.
        :param pre_tkns (torch.Tensor, Nx512): Learnable pre-tokens (kind of like positional embeddings)

        :return tkns (torch.Tensor, Nx512): N separate image-dependent tokens.
        """
        y = torch.concat(pre_tkns, dim=0)
        x = img_f_agg.expand(-1, y.shape[0], -1)
        y = y.unsqueeze(0).expand(img_f_agg.shape[0], -1, -1)
        x = torch.concat((x, y), dim=2)     # x.shape = BxNx(C+512)
        tkns = self.img_f_agg2tkndim(x)     # tkns.shape = BxNx512

        return tkns


class DepthCLIP(nn.Module):
    """
    A semi-faithful implementation of DepthCLIP, with various modifications.
    Based heavily on DepthCLIP from "Can Language Understand Depth?", Zhang et al., 2022, in ACM Multimedia 2022.
    The general idea is to align depth-related language and image features using a pretrained CLIP model.
    This implementation does not exactly follow that in the paper, and also contains various modifications and experiments.
    """
    def __init__(self, 
                 args, 
                 features=256, 
                 out_channels=[256, 512, 1024, 1024], 
                 use_bn=False, 
                 use_clstoken=False, 
                 localhub=True
                 ):
        super().__init__()
        self.args = args

        self._encoder_params_module_list = []
        self._non_encoder_params_module_list = []
        self._frozen_params_module_list = []
        self._zero_params_module_list = []  # Used for CLIP. Requires_grad=True but not passed to optimizer, so shouldn't update...
        
        self._extra_learnable_params_list = []  # Temporary holding place for learnable tokens. If self.args.depthclip.freeze_depthclip is True, these will be frozen. Else they will be learnable.


        if self.args.basic.dataset == 'tusimple':
            self.ReturnType = namedtuple('ReturnType', ['depth_pred', 'depth_logits', 'dense_features', 'text_features', 'concatenated_depth_pred','concatenated_depth_gt','rank_logits'])
        else:
            self.ReturnType = namedtuple('ReturnType', ['depth_pred', 'depth_logits', 'dense_features', 'text_features'])
            
        # self.ReturnType = namedtuple('ReturnType', ['depth_pred', 'depth_logits', 'dense_features', 'text_features'])

        self.temperature = 0.1

        lang_strat = self.args.depthclip.lang_strat


        # self.conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        # nn.init.zeros_(self.conv_layer.weight)
        # nn.init.zeros_(self.conv_layer.bias)
        # nn.init.normal_(self.conv_layer.weight)
        # nn.init.normal_(self.conv_layer.bias)
        # self.relu = nn.ReLU()

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
        #### Language strategy parsing

        # Default for extra tokens to add to the tokenizer vocabulary
        # Will get overridden whenever there are learnable tokens to be added. Each token is replaced with its learned counterpart
        # as part of the tokenization process (when calling clip.tokenize(text, extra_tkns)).
        self.extra_tkns_cfg = {}    # Contains k:v pairs of token name, number of extra tokens with that name.
        self.extra_tkns_learnables = {}   # Contains refs to extra LEARNABLE tokens, if any.  
        self.extra_tkns_lookup = {}   # Key = new token, value = learnable parameter for that token.
        self.extra_tkns_reverse_lookup = {}   # Key = token index, value = learnable parameter for that token.

        # Templates to use
        if lang_strat.get("templates") == 'paper':
            # Static templates (i.e. the template words are static, but the depth/object tokens can be either.)
                self.sentence_templates = ["This {object_tkn} is {depth_tkn}"]
        elif lang_strat.get("templates") == 'paper-v1':
                self.sentence_templates = ["This {object_tkn} appears to be {depth_tkn}"]
            
            # Learnable templates. <x>o<y>d means x learnable tokens, then the object token, then y more learnable tokens, then the depth token.
        elif lang_strat.get("templates") == "learned-static-1o1d":
                self.sentence_templates = ["<|prompt_0|> {object_tkn} <|prompt_1|> {depth_tkn}"]
                self.extra_tkns_cfg["prompt"] = 2
                self.extra_tkns_learnables_prompt = nn.ParameterList(values=[torch.normal(mean=0, std=0.02, size=(1, 512)) for i in range(2)])
                self._extra_learnable_params_list += self.extra_tkns_learnables_prompt
                for i in range(2):
                    self.extra_tkns_lookup[f"<|prompt_{i}|>"] = self.extra_tkns_learnables_prompt[i]
        elif lang_strat.get("templates") == "learned-static-1o2d":
                self.sentence_templates = ["<|prompt_0|> {object_tkn} <|prompt_1|> <|prompt_2|> {depth_tkn}"]
                self.extra_tkns_cfg["prompt"] = 3
                self.extra_tkns_learnables_prompt = nn.ParameterList(values=[torch.normal(mean=0, std=0.02, size=(1, 512)) for i in range(3)])
                self._extra_learnable_params_list += self.extra_tkns_learnables_prompt
                for i in range(3):
                    self.extra_tkns_lookup[f"<|prompt_{i}|>"] = self.extra_tkns_learnables_prompt[i]
        elif lang_strat.get("templates") == "learned-static-4o4d":
                self.sentence_templates = ["<|prompt_0|> <|prompt_1|> <|prompt_2|> <|prompt_3|> {object_tkn} <|prompt_4|> <|prompt_5|> <|prompt_6|> <|prompt_7|> {depth_tkn}"]
                self.extra_tkns_cfg["prompt"] = 8
                self.extra_tkns_learnables_prompt = nn.ParameterList(values=[torch.normal(mean=0, std=0.02, size=(1, 512)) for i in range(8)])
                self._extra_learnable_params_list += self.extra_tkns_learnables_prompt
                for i in range(8):
                    self.extra_tkns_lookup[f"<|prompt_{i}|>"] = self.extra_tkns_learnables_prompt[i]
        else:
                sys.exit("Error: sentence templates set not recognised.")

        # Depth tokens to use
        # Depth tokens to use
        if lang_strat.get("depth_tokens") == 'paper':
                self.depth_tokens = [
                    "giant",
                    "extremely close",
                    "close",
                    "not in distance",
                    "a little remote",
                    "far",
                    "unseen"
                ]
        elif lang_strat.get("depth_tokens") == "size-7":
                self.depth_tokens = [
                    "very small",
                    "small",
                    "slightly small",
                    "neither small nor large",
                    "slightly large",
                    "large",
                    "very large",
                ]
        elif lang_strat.get("depth_tokens") == "depth-7":
                self.depth_tokens = [
                    "very close",
                    "close",
                    "slightly close",
                    "neither close nor distant",
                    "slightly distant",
                    "distant",
                    "very distant"
                ]
        elif lang_strat.get("depth_tokens") == "colour-7":
                self.depth_tokens = [
                    "very red",
                    "red",
                    "slightly red",
                    "neither red nor green",
                    "slightly green",
                    "green",
                    "very green"
                ]
            # Learned depth tokens
        elif lang_strat.get("depth_tokens") == "learned-static-7":
                # 7 learned depth tokens. NOT input-dependent: represented as learnable parameters.
                self.depth_tokens = [f"<|depth_{i}|>" for i in range(7)]
                self.extra_tkns_cfg["depth"] = 7
                self.extra_tkns_learnables_depth = nn.ParameterList(values=[torch.normal(mean=0, std=0.02, size=(1, 512)) for i in range(7)])
                self._extra_learnable_params_list += self.extra_tkns_learnables_depth
                for i in range(7):
                    self.extra_tkns_lookup[f"<|depth_{i}|>"] = self.extra_tkns_learnables_depth[i]
            # Learned depth tokens
        elif lang_strat.get("depth_tokens") == "learned-static-20":
                # 20 learned depth tokens. NOT input-dependent: represented as learnable parameters.
                self.depth_tokens = [f"<|depth_{i}|>" for i in range(20)]
                self.extra_tkns_cfg["depth"] = 20
                self.extra_tkns_learnables_depth = nn.ParameterList(values=[torch.normal(mean=0, std=0.02, size=(1, 512)) for i in range(20)])
                self._extra_learnable_params_list += self.extra_tkns_learnables_depth
                for i in range(20):
                    self.extra_tkns_lookup[f"<|depth_{i}|>"] = self.extra_tkns_learnables_depth[i]
        elif lang_strat.get("depth_tokens") == "learned-static-128":
                # 128 learned depth tokens. NOT input-dependent: represented as learnable parameters.
                self.depth_tokens = [f"<|depth_{i}|>" for i in range(128)]
                self.extra_tkns_cfg["depth"] = 128
                self.extra_tkns_learnables_depth = nn.ParameterList(values=[torch.normal(mean=0, std=0.02, size=(1, 512)) for i in range(128)])
                self._extra_learnable_params_list += self.extra_tkns_learnables_depth
                for i in range(128):
                    self.extra_tkns_lookup[f"<|depth_{i}|>"] = self.extra_tkns_learnables_depth[i]
        elif lang_strat.get("depth_tokens") == "learned-static-256":
                # 256 learned depth tokens. NOT input-dependent: represented as learnable parameters.
                self.depth_tokens = [f"<|depth_{i}|>" for i in range(256)]
                self.extra_tkns_cfg["depth"] = 256
                self.extra_tkns_learnables_depth = nn.ParameterList(values=[torch.normal(mean=0, std=0.02, size=(1, 512)) for i in range(256)])
                self._extra_learnable_params_list += self.extra_tkns_learnables_depth
                for i in range(256):
                    self.extra_tkns_lookup[f"<|depth_{i}|>"] = self.extra_tkns_learnables_depth[i]
        else:
                sys.exit("Error: language strategy depth tokens set not recognised")

        # Object tokens to use
        if lang_strat.get("object_tokens") == 'paper':
            # Static object tokens
                self.object_tokens = ["patch"]
        elif lang_strat.get("object_tokens") == "learned-static-1":
                # 1 learned object token. NOT input-dependent: represented as learnable parameters.
                self.object_tokens = [f"<|object_{i}|>" for i in range(1)]
                self.extra_tkns_cfg["object"] = 1
                self.extra_tkns_learnables_object = nn.ParameterList(values=[torch.normal(mean=0, std=0.02, size=(1, 512))])
                self._extra_learnable_params_list += self.extra_tkns_learnables_object
                for i in range(1):
                    self.extra_tkns_lookup[f"<|object_{i}|>"] = self.extra_tkns_learnables_object[i]
        elif lang_strat.get("object_tokens") == "imgf":
                # Uses an img_f (after running through a transformation) as the token.
                self.object_tokens = ["<|imgf_0|>"]
                self.extra_tkns_cfg["imgf"] = 1
                self.extra_tkns_lookup["<|imgf_0|>"] = None   # This gets overwritten in the embedding loop (since it's different every time, a constant name doesn't make sense)
        else:
                sys.exit("Error: language strategy object tokens set not recognised")
            
        # Depth bin centres to use
        if lang_strat.get("depth_bin_centres") == 'paper':
                self.depth_bin_centres = [1.00, 1.50, 2.00, 2.25, 2.50, 2.75, 3.00]
                self.bins = len(self.depth_tokens)
        elif lang_strat.get("depth_bin_centres") == "dset-even-7":
                # Evenly-spaced across the whole range of the dataset being used.
                dset_range = args[args.basic.dataset].max_depth - args[args.basic.dataset].min_depth
                bin_width = dset_range / 7
                self.depth_bin_centres = [args[args.basic.dataset].min_depth + (i * (bin_width)) + (bin_width / 2) for i in range(7)]
                self.bins = len(self.depth_tokens)
        elif lang_strat.get("depth_bin_centres") == "dset-log-7":
                # Evenly-spaced across the whole range of the dataset being used.
                dset_range = args[args.basic.dataset].max_depth - args[args.basic.dataset].min_depth
                dset_range_log = math.log(args[args.basic.dataset].max_depth) - math.log(args[args.basic.dataset].min_depth)
                bin_width_log = dset_range_log / 7
                self.depth_bin_centres_log = [math.log(args[args.basic.dataset].min_depth) + (i * (bin_width_log)) + (bin_width_log / 2) for i in range(7)]
                self.depth_bin_centres = [math.e**i for i in self.depth_bin_centres_log]
                self.bins = len(self.depth_tokens)
        elif lang_strat.get("depth_bin_centres") == "const80-log-7":
                # Evenly-spaced across the range 0.001-80
                max_depth = 80
                min_depth = 0.001
                dset_range = max_depth - min_depth
                dset_range_log = math.log(max_depth) - math.log(min_depth)
                bin_width_log = dset_range_log / 7
                self.depth_bin_centres_log = [math.log(min_depth) + (i * (bin_width_log)) + (bin_width_log / 2) for i in range(7)]
                self.depth_bin_centres = [math.e**i for i in self.depth_bin_centres_log]
                self.bins = len(self.depth_tokens)
        elif lang_strat.get("depth_bin_centres") == "dset-even-20":
                # Evenly-spaced across the whole range of the dataset being used.
                dset_range = args[args.basic.dataset].max_depth - args[args.basic.dataset].min_depth
                bin_width = dset_range / 20
                self.depth_bin_centres = [args[args.basic.dataset].min_depth + (i * (bin_width)) + (bin_width / 2) for i in range(20)]
                self.bins = len(self.depth_tokens)
        elif lang_strat.get("depth_bin_centres") == "dset-log-20":
                # Evenly-spaced across the whole range of the dataset being used.
                dset_range = args[args.basic.dataset].max_depth - args[args.basic.dataset].min_depth
                dset_range_log = math.log(args[args.basic.dataset].max_depth) - math.log(args[args.basic.dataset].min_depth)
                bin_width_log = dset_range_log / 20
                self.depth_bin_centres_log = [math.log(args[args.basic.dataset].min_depth) + (i * (bin_width_log)) + (bin_width_log / 2) for i in range(20)]
                self.depth_bin_centres = [math.e**i for i in self.depth_bin_centres_log]
                self.bins = len(self.depth_tokens)
        elif lang_strat.get("depth_bin_centres") == "const80-log-20":
                # Evenly-spaced across the range 0.001-80
                max_depth = 80
                min_depth = 0.001
                dset_range = max_depth - min_depth
                dset_range_log = math.log(max_depth) - math.log(min_depth)
                bin_width_log = dset_range_log / 20
                self.depth_bin_centres_log = [math.log(min_depth) + (i * (bin_width_log)) + (bin_width_log / 2) for i in range(20)]
                self.depth_bin_centres = [math.e**i for i in self.depth_bin_centres_log]
                self.bins = len(self.depth_tokens)
        elif lang_strat.get("depth_bin_centres") == "dset-even-128":
                # Evenly-spaced across the whole range of the dataset being used.
                dset_range = args[args.basic.dataset].max_depth - args[args.basic.dataset].min_depth
                bin_width = dset_range / 128
                self.depth_bin_centres = [args[args.basic.dataset].min_depth + (i * (bin_width)) + (bin_width / 2) for i in range(128)]
                self.bins = len(self.depth_tokens)
        elif lang_strat.get("depth_bin_centres") == "dset-even-256":
                # Evenly-spaced across the whole range of the dataset being used.
                dset_range = args[args.basic.dataset].max_depth - args[args.basic.dataset].min_depth
                bin_width = dset_range / 256
                self.depth_bin_centres = [args[args.basic.dataset].min_depth + (i * (bin_width)) + (bin_width / 2) for i in range(256)]
                self.bins = len(self.depth_tokens)
        elif lang_strat.get("depth_bin_centres") == "dset-log-256":
                # Evenly-spaced across the whole range of the dataset being used.
                dset_range = args[args.basic.dataset].max_depth - args[args.basic.dataset].min_depth
                dset_range_log = math.log(args[args.basic.dataset].max_depth) - math.log(args[args.basic.dataset].min_depth)
                bin_width_log = dset_range_log / 256
                self.depth_bin_centres_log = [math.log(args[args.basic.dataset].min_depth) + (i * (bin_width_log)) + (bin_width_log / 2) for i in range(256)]
                self.depth_bin_centres = [math.e**i for i in self.depth_bin_centres_log]
                self.bins = len(self.depth_tokens)
        elif lang_strat.get("depth_bin_centres") == "const80-log-256":
                # Evenly-spaced across the range 0.001-80
                max_depth = 80
                min_depth = 0.001
                dset_range = max_depth - min_depth
                dset_range_log = math.log(max_depth) - math.log(min_depth)
                bin_width_log = dset_range_log / 256
                self.depth_bin_centres_log = [math.log(min_depth) + (i * (bin_width_log)) + (bin_width_log / 2) for i in range(256)]
                self.depth_bin_centres = [math.e**i for i in self.depth_bin_centres_log]
                self.bins = len(self.depth_tokens)
        else:
                sys.exit("Error: language strategy depth bin centres set not recognised")

        if "RN" in self.args.depthclip.clip:
            img_f_agg_dims = 2048
        elif "ViT" in self.args.depthclip.clip:
            img_f_agg_dims = 512
        else:
            sys.exit("Error in DepthCLIP init: img_f_agg_dims unknown for this CLIP architecture.")

        if self.args.depthclip.lang_strat.get("img_f_dependent") == 'basic-1':
                self.img_f_2_txt_tkns = Img_f_agg2tkns(img_f_agg_dims, 512)
                self._non_encoder_params_module_list.append(self.img_f_2_txt_tkns)   # Extra tokens are learnable.
        else:
                pass

        # Sanity checking
        assert self.sentence_templates is not None
        assert self.depth_tokens is not None
        assert self.object_tokens is not None
        assert self.depth_bin_centres is not None



        # Initialise the output stages:
        # Arch modifies three parts:
        #   1. Img features, before text correlation
        #   2. Bin probabilities (after text correlation)
        #   3. Depth map output

        # Initialise CLIP
        self.clip, self.clip_preprocess = clip.load(self.args.depthclip.clip, device="cpu", download_root='checkpoints')
        _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _dinov2_path = os.environ.get('DINOV2_PRETRAIN_PATH', os.path.join(_repo_root, 'checkpoints', 'dinov2_vitb14_pretrain.pth'))
        self.dinov2:nn.Module = torch.hub.load('dinov2', 'dinov2_vitb14', source='local', verbose=True, pretrained=False)
        self.dinov2.load_state_dict(torch.load(_dinov2_path, map_location='cpu'))

        # Optionally, load a checkpoint.
        if self.args.depthclip.get("start_from_checkpoint"):
            ckpt_path = self.args.depthclip.get("start_from_checkpoint")
            tmp = torch.load(os.path.expanduser(ckpt_path))
            pattern = re.compile("model.*")
            # Overwrite weight names to work from here (they're saved from a level up)
            tmp_state_dict = {re.sub(r"model\.", "", k): v for k, v in tmp["state_dict"].items() if pattern.match(k)}
            # extra_tkns_learnables used to be nn.Embeddings, now they're just nn.Parameters. This allows loading of the old embeddings checkpoints.
            tmp_state_dict = {re.sub(r"\.embeddings\.weight", "", k) if re.compile("extra_tkns_learnables_.*\.embeddings\.weight").match(k) else k: v for k, v in tmp_state_dict.items()}

            if self.args.depthclip.get("load_clip_from_checkpoint") is False:
                # Filter all CLIP values from the state dict
                tmp_state_dict = {k: v for k, v in tmp_state_dict.items() if re.compile("(?!^clip.*)").match(k)}
                

            self.load_state_dict(tmp_state_dict, strict=False)

        # Freeze (requires_grad = False) all of CLIP that can be frozen
        # self._encoder_params_module_list.append(self.clip.visual)
        # self._extra_learnable_params_list.append(self.clip.visual)
        # self._extra_learnable_params_list.append(self.transposed_conv_layer)
        # self._extra_learnable_params_list.append(self.conv_layer)
        
        # self._encoder_params_module_list.append(self.clip.visual)
        self._encoder_params_module_list.append(self.clip.visual)
        self._frozen_params_module_list.append(self.clip.token_embedding)
        self._frozen_params_module_list.append(nn.ParameterList([self.clip.positional_embedding]))
        self._frozen_params_module_list.append(nn.ParameterList([self.clip.text_projection]))
        self._frozen_params_module_list.append(nn.ParameterList([self.clip.logit_scale]))
        self._frozen_params_module_list.append(self.clip.transformer)
        self._frozen_params_module_list.append(self.clip.ln_final)




        if self.args.depthclip.get("freeze_clip") == True:
                self._encoder_params_module_list.append(self.clip.visual)
                # self._encoder_params_module_list.append(self.transposed_conv_layer)
                # self._encoder_params_module_list.append(self.conv_layer)
                # Completely freeze everything in depthclip. Will not update tokens either.
                self._frozen_params_module_list.append(nn.ParameterList([self.clip.positional_embedding]))
                self._frozen_params_module_list.append(nn.ParameterList([self.clip.text_projection]))
                self._frozen_params_module_list.append(nn.ParameterList([self.clip.logit_scale]))
                self._frozen_params_module_list.append(self.clip.transformer)
                self._frozen_params_module_list.append(self.clip.ln_final)
                self._frozen_params_module_list += self._extra_learnable_params_list    # Extra tokens are also frozen.
        else:
                # Zero-out (don't pass to optimizer) all other params
                self._zero_params_module_list.append(nn.ParameterList([self.clip.positional_embedding]))
                self._zero_params_module_list.append(nn.ParameterList([self.clip.text_projection]))
                self._zero_params_module_list.append(nn.ParameterList([self.clip.logit_scale]))
                self._zero_params_module_list.append(self.clip.transformer)
                self._zero_params_module_list.append(self.clip.ln_final)
                self._non_encoder_params_module_list += self._extra_learnable_params_list   # Extra tokens are learnable.

        # Build templates (word-level). Doesn't handle img_f substitution.
        self.texts = []
        for template in self.sentence_templates:
            for obj in self.object_tokens:
                for depth in self.depth_tokens:
                    self.texts.append(template.format(object_tkn=obj, depth_tkn=depth))

        # Convert all tokens to integer indices. New (out-of-vocab) tokens will have large nonzero values and will be in extra_tkns_reverse_lookup.
        # These get reused, so they're initialised here.
        self.texts_tokenized, self.tokenizer = clip.tokenize(self.texts, extra_tkns_cfg=self.extra_tkns_cfg)  # tokenize
        
        # the dimension of feature from CLIP visual encoder
        dino_dim = 768 
        clip_aligned_dim = 256
        self.use_clip = args.model.get('use_clip', True)
        self.use_dino = args.model.get('use_dino', True)
        self.freeze_dino = args.model.get('freeze_dino', False)

        # the decoder
        dpt_in_channels = 0
        if self.use_clip:
            dpt_in_channels += clip_aligned_dim
        if self.use_dino:
            dpt_in_channels += dino_dim
        self.depth_head = DPTHead(1, dpt_in_channels, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

        # fuse clip and dino
        self.dino_clip_align_method = self.args.model.get('dino_clip_align_method', 'interp_conv1x1')
        if self.dino_clip_align_method == 'interp_conv1x1':
            self.clip_align_conv = nn.ModuleList([
                nn.Identity(),
                nn.Conv2d(512, clip_aligned_dim, 1, 1),
                nn.Conv2d(1024, clip_aligned_dim, 1, 1),
                nn.Conv2d(2048, clip_aligned_dim, 1, 1)
            ])
        elif self.dino_clip_align_method == 'convs2_interp':
            self.clip_align_conv = nn.ModuleList([
                nn.Conv2d(256, clip_aligned_dim, 3, 2, 1),
                nn.Conv2d(512, clip_aligned_dim, 3, 2, 1),
                nn.Conv2d(1024, clip_aligned_dim, 1, 1),
                nn.ConvTranspose2d(2048, clip_aligned_dim, 3, 2, 1)
            ])
        else:
            raise NotImplementedError

        self._non_encoder_params_module_list.append(self.depth_head)
        self._non_encoder_params_module_list.append(self.clip_align_conv)

        # freeze dino's params
        if self.use_dino:
            if self.freeze_dino:
                for params in self.dinov2.parameters():
                    params.requires_grad_(False)
            else:
                self._encoder_params_module_list.append(self.dinov2)

    def get_encoder_params(self):
        for m in self._encoder_params_module_list:
            if isinstance(m, nn.Parameter):
                yield m    # must return a generator
            else:
                yield from m.parameters()


    def get_non_encoder_params(self):
        for m in self._non_encoder_params_module_list:
            if isinstance(m, nn.Parameter):
                yield m    # must return a generator
            else:
                yield from m.parameters()

    
    def get_frozen_params(self):
        for m in self._frozen_params_module_list:
            if isinstance(m, nn.Parameter):
                yield m    # must return a generator
            else:
                yield from m.parameters()


    def get_zero_params(self):
        for m in self._zero_params_module_list:
            if isinstance(m, nn.Parameter):
                yield m    # must return a generator
            else:
                yield from m.parameters()


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

    def fuse_dino_clip(self, middle_feats_dino, middle_feats_clip):
        assert len(middle_feats_clip) == len(middle_feats_dino)
        fused_feats = []
        for i, (feat_clip, feat_dino) in enumerate(zip(middle_feats_clip, middle_feats_dino)):
            # 方法一：通过插值对齐大小，再通过1x1卷积对齐特征维度
            if self.dino_clip_align_method == 'interp_conv1x1':
                aligned_feat_clip = F.interpolate(feat_clip, feat_dino.shape[-2:])
                aligned_feat_clip = self.clip_align_conv[i](aligned_feat_clip)

            # 方法二：通过下/上采样卷积尽量对齐大小和通道维度，再通过插值完全对齐大小
            elif self.dino_clip_align_method == 'convs2_interp':
                aligned_feat_clip = self.clip_align_conv[i](feat_clip)
                aligned_feat_clip = F.interpolate(aligned_feat_clip, feat_dino.shape[-2:])
            else:
                raise NotImplementedError

            fused_feat = torch.concatenate([feat_dino, aligned_feat_clip], dim=1)
            #fused_feat = feat_dino
            fused_feats.append(fused_feat)
        return fused_feats

    def forward(self, image, column, row, img_shape_list, resize_ratios):
        self.in_height = image.shape[2]
        self.in_width = image.shape[3]
        patch_h = 14
        patch_w = 14
        out_height_dino = int(self.in_height / patch_h)
        out_width_dino = int(self.in_width / patch_w)
        if "ViT" in self.args.depthclip.clip:
            out_height = int(self.in_height / self.clip.visual.patch_size)
            out_width = int(self.in_width / self.clip.visual.patch_size)        
        elif "RN50" in self.args.depthclip.clip:
            out_height = int(self.in_height / 32)
            out_width = int(self.in_width / 32)
        else:
            pass

        # Run through most of the CLIP encoder to get dense CLIP features
        ## For RN50 backbone, C = 2048
        ## For ViT-B/32 backbone, C = 512
        img_f_tkns = None


        if self.args.basic.dataset == 'tusimple':
            #image_features_dino = self.dinov2.forward_features(image, img_shape_list=img_shape_list)['x_norm_patchtokens']
            if self.use_dino:
                middle_feats_dino = self.dinov2.get_intermediate_layers(image, [2, 5, 8, 11], return_class_token=True, img_shape_list=img_shape_list, reshape=True)
                middle_feats_dino, middle_dino_cls_tokens = list(zip(*middle_feats_dino))
            else:
                middle_feats_dino = None
            # image_features_dino = middle_feats_dino[-1][:, 1:]
            # image_features_dino = image_features_dino.view(image_features_dino.shape[0], out_height_dino, out_width_dino, image_features_dino.shape[-1]).permute(0, 3, 1, 2)
            if self.use_clip:
                image_features_clip, depth_x, middle_feats_clip = self.clip.encode_image(image,rank=False,csa=False,return_middle=True)
                middle_feats_clip = list(middle_feats_clip.values())
            else:
                middle_feats_clip = None
        else:
            _, depth_x = self.clip.encode_image(image, csa=True)   # B, HW, C where H and W are patch coordinates not pixel coordinates
        
        # TODO: 怎么把dino和clip(多层)的feature map融合起来，大小和通道不一样
        if self.use_clip and self.use_dino:
            fused_feats = self.fuse_dino_clip(middle_feats_dino, middle_feats_clip)
        elif self.use_dino:
            fused_feats = middle_feats_dino
        elif self.use_clip:
            fused_feats = []
            for i, feat_clip in enumerate(middle_feats_clip):
                aligned_feat_clip = F.interpolate(feat_clip, [out_height_dino, out_width_dino])
                aligned_feat_clip = self.clip_align_conv[i](aligned_feat_clip)
                fused_feats.append(aligned_feat_clip)

        #img_f = depth_x[-1]/depth_x[-1].norm(dim=-1, keepdim=True) 
        # img_f = img_f / img_f.norm(dim=-1, keepdim=True) 
        # add the ranking 
        num_regions = 7
        region_size = 32
        #image_feature_process = img_f.permute(0, 2, 1).reshape((img_f.shape[0], img_f.shape[-1], patch_h, patch_w))
        selected_patches_clip = []
        selected_patches_dino = []
        selected_patches_fusion = []
        selected_patches_coords = []
        use_roi_align = self.args.model.get('use_roi_align', False)
        for id in range(image.shape[0]):
            if row[id] is None:
                 continue
            row_num = len(row[id])
            column_id = column[id]
            row_id = row[id]
            selected_patches_clip_all_lanes = []
            selected_patches_dino_all_lanes = []
            selected_patches_fusion_all_lanes = []
            selected_patches_coords_all_lanes = []
            for i in range(row_num):
                x_filtered = [val for val in row_id[i] if val >= 0]
                y_filtered = [column_id[j] for j, val in enumerate(row_id[i]) if val >= 0]
                num_points = len(x_filtered)
                interval = (num_points - 1) // (num_regions - 1)
                if len(x_filtered) < num_regions:
                    pass
                else:
                    selected_patches_clip_single_lane = []
                    selected_patches_dino_single_lane = []
                    selected_patches_fusion_single_lane = []
                    selected_patches_coords_single_lanes = []
                    for j in range(num_regions):
                        index = j * interval
                        x = x_filtered[index]
                        y = y_filtered[index]
                        multiscale_croped_feats = []
                        for k in range(len(fused_feats)):
                            if use_roi_align:
                                x_min = max(0, x - region_size // 2)
                                y_min = max(0, y - region_size // 2)
                                x_max = min(img_shape_list[id][1]/resize_ratios[id], x + region_size // 2)
                                y_max = min(img_shape_list[id][0]/resize_ratios[id], y + region_size // 2)
                                croped_box = torch.Tensor([x_min, y_min, x_max, y_max]).unsqueeze(0).cuda()
                                croped_feat = torchvision.ops.roi_align(fused_feats[k][id].unsqueeze(0), boxes=[croped_box], spatial_scale=resize_ratios[id]/14, output_size=7)
                                croped_feat = torch.nn.functional.avg_pool2d(croped_feat, 7).squeeze()
                            else:
                                x_sample = int(x*resize_ratios[id]/patch_w)
                                y_sample = int(y*resize_ratios[id]/patch_h)
                                x_sample = min(fused_feats[k].shape[-1]-1, x_sample)
                                y_sample = min(fused_feats[k].shape[-2]-1, y_sample)
                                croped_feat = fused_feats[k][id, :, y_sample, x_sample]
                            multiscale_croped_feats.append(croped_feat)
                        multiscale_croped_feats = torch.concatenate(multiscale_croped_feats)
                        selected_patches_fusion_single_lane.append(multiscale_croped_feats)
                        selected_patches_coords_single_lanes.append([x.item(), y.item()])
                        # selected_patches_fusion_single_lane.append(torch.concatenate([croped_dino_feat, croped_clip_feat]))
                        #selected_patches_clip_single_lane.append(multiscale_croped_clip_feats)

                    #selected_patches_clip_all_lanes.append(selected_patches_clip_single_lane)
                    #selected_patches_dino_all_lanes.append(selected_patches_dino_single_lane)
                    selected_patches_fusion_all_lanes.append(torch.stack(selected_patches_fusion_single_lane))
                    selected_patches_coords_all_lanes.append(selected_patches_coords_single_lanes)
            #selected_patches_clip.append(selected_patches_clip_all_lanes)
            #selected_patches_dino.append(selected_patches_dino_all_lanes)
            selected_patches_fusion.append(selected_patches_fusion_all_lanes)
            selected_patches_coords.append(selected_patches_coords_all_lanes)
        batch_if = selected_patches_fusion
                
        txt_f = self.get_language_features(img_f_tkns = None, extra_tkns_override = None)      # 3 dims
        
        # Interpolate image feature dim to match text feature dim.
        # if img_f_tkns is not None:
        #     # In this case, txt_f will be BxHWxCxN (One (set of) txt_f for each patch feature in img_f)
        #     img_f = F.interpolate(img_f, size=txt_f.shape[2])
        # elif len(txt_f.shape) == 3:
        #     img_f = F.interpolate(img_f, size=txt_f.shape[1])
        # else:
        #     # In this case, txt_f will be CxN (One (set of) txt_f.)
        #     img_f = F.interpolate(img_f, size=txt_f.shape[0])

 
        # How to do the correlation between language and image features.
        # Default is with @ operator (torch.mm internally), img_f is BxHWxK, txt_f is KxN where N is number of prompts.
        if self.args.depthclip.get("lang_corr_stage") == 'depthclip':# Default behaviour is the same as depthclip behaviour. Done for backwards compatibility.
                if self.args.basic.dataset == 'tusimple':
                    contrastive_loss = torch.tensor(0.0, dtype=torch.float32).to(txt_f.device)
                    logits = []
                    lane_cnt = 0
                    for id in range(len(batch_if)):
                        logits_per_img = []
                        for num in range(len(batch_if[id])):
                            im_feature = batch_if[id][num] # every lane should have 7 patch 
                            # TODO: MLP降维？or 插值
                            im_feature = F.interpolate(im_feature.unsqueeze(0), size=txt_f.shape[0]).squeeze()
                            im_feature = im_feature / (im_feature.norm(dim=-1, keepdim=True)+1e-6)

                            depth_logits = 100.0 * im_feature @ txt_f
                            logits_per_img.append(depth_logits)
                            
                            # single modal contrastive for image
                            #contrastive_loss = 100*self.contrastive_loss(im_feature)
                            contrastive_loss += self.contrastive_loss(im_feature)
                            lane_cnt += 1
                        logits.append(logits_per_img)
                    if lane_cnt > 0:
                        contrastive_loss /= lane_cnt
                    # TODO: 怎么解码深度？
                    # return logits, contrastive_loss
                    # depth = self.depth_head(depth_x[:-1]+ [img_f], patch_h, patch_w)
                    # depth = F.interpolate(depth, size=(patch_h, patch_w), mode="bilinear", align_corners=True)
                    # depth = F.relu(depth)

                    kitti_mask = torch.Tensor([True if item is None else False for item in column]).bool()
                    fused_feats_kitti = []
                    for k in range(len(fused_feats)):
                        fused_feats_kitti.append(fused_feats[k][kitti_mask])
                    depth = self.depth_head(fused_feats_kitti, out_height_dino, out_width_dino)
                    #depth = F.relu(depth) # dpt head输出的时候已经relu过了，这里应该不需要

                    return logits, contrastive_loss, depth, selected_patches_coords
                    # return logits, depth.squeeze(1)                    
    
                else:    
                    depth_logits = 100.0 * (img_f @ txt_f)  # B, HW, K # img_f and text_f have both been normalized, so just use a inner product
                    depth_logits = depth_logits.permute(0, 2, 1).reshape(image.shape[0], self.bins, out_height, out_width)  # B, K, H, W 
                    depth_logits /= self.temperature

                    # Depth logit post-processing
                    depth_logits = self.post_lang_corr_stage(depth_logits)

                    depth_pred = F.softmax(depth_logits, dim=1)
                    bin_tensor = torch.tensor(self.depth_bin_centres).to(depth_pred.device)
                    depth_pred = depth_pred * bin_tensor.reshape(1, self.bins).unsqueeze(-1).unsqueeze(-1)
                    depth_pred = depth_pred.sum(1, keepdim=True)


                    # depth_pred_conv  = (self.conv_layer(self.relu(self.transposed_conv_layer(depth_pred))))
                    depth_pred_inter = nn.functional.interpolate(depth_pred, depth_pred_conv.shape[-2:], mode='bilinear', align_corners=True) 
                    depth_pred = depth_pred_inter + depth_pred_conv

                    # print(self.conv_layer.weight)



                    # Prediction post-processing
                    depth_pred = self.post_depth_pred_stage(depth_pred)
                    return self.ReturnType(depth_pred=depth_pred, depth_logits=F.softmax(depth_logits), dense_features=img_f, text_features=txt_f)
                
                
                
    def contrastive_loss(self, image_feature):
        n = image_feature.size(0)  # 获取向量的数量
        dim = image_feature.size(1)  # 获取向量的维度

        # 计算余弦相似度矩阵
        A_norm = F.normalize(image_feature, p=2, dim=1)  # 将向量归一化
        cosine_similarity_matrix = torch.mm(A_norm, A_norm.t())  # 计算余弦相似度

        # 生成索引差异矩阵
        index_matrix = torch.arange(n).view(-1, 1) - torch.arange(n).view(1, -1)
        distance_matrix = index_matrix.abs().float().to(cosine_similarity_matrix.device)

        # 生成权重矩阵，索引差异越大，权重越大
        weight_matrix = distance_matrix / distance_matrix.max()
        # weight_matrix = (torch.exp(0.5*distance_matrix)-1) / (torch.exp(0.5*distance_matrix)-1).max()

        # 计算对比损失
        loss = (weight_matrix * cosine_similarity_matrix).sum() / (n * n)

        return loss