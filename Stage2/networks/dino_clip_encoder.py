# DepthCLIP.py
# Based heavily on DepthCLIP from "Can Language Understand Depth?", Zhang et al., 2022, in ACM Multimedia 2022
# The general idea is to align depth-related language and image features using a pretrained CLIP model.
# This implementation does not exactly follow that in the paper, and also contains various modifications and experiments.

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
from layers import *

from CLIP import clip
import sys
# from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
# from loralib.run_utils import *
class depthdinoclipencoder(nn.Module):
    """
    A semi-faithful implementation of DepthCLIP, with various modifications.
    Based heavily on DepthCLIP from "Can Language Understand Depth?", Zhang et al., 2022, in ACM Multimedia 2022.
    The general idea is to align depth-related language and image features using a pretrained CLIP model.
    This implementation does not exactly follow that in the paper, and also contains various modifications and experiments.
    """
    def resnet50_forward(self, x):
        # See note [TorchScript super()]
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        stage1 = self.resnet50.layer1(x)
        stage2 = self.resnet50.layer2(stage1)
        stage3 = self.resnet50.layer3(stage2)
        stage4 = self.resnet50.layer4(stage3)
        return [stage1, stage2, stage3, stage4]

    def __init__(self, args, pose=False):
        super().__init__()
        self.args = args
        self.use_resnet50_instead_clip = args.use_resnet50_instead_clip
        self.dinov2:nn.Module = torch.hub.load('/code/CFMDE-main/dinov2', 'dinov2_vitb14', source='local', verbose=True, pretrained=False)
        self.dinov2.load_state_dict(torch.load('/code/CFMDE-main/checkpoints/dinov2_vitb14_pretrain.pth'))
        if self.use_resnet50_instead_clip:
            self.resnet50 = resnet50()
            if not args.resnet50_random_init:
                sd = torch.load('/code/CFMDE-main/resnet50-0676ba61.pth')
                self.resnet50.load_state_dict(sd)
            del self.resnet50.avgpool
            del self.resnet50.fc
        else:
            self.clip, self.clip_preprocess = clip.load('RN50', device="cpu", download_root='/code/CFMDE-main/checkpoints')

        clip_aligned_dim = 256
        # fuse clip and dino
        self.dino_clip_align_method = self.args.dino_clip_align_method
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
        self.use_clip = not args.only_dino
        self.only_clip = args.only_clip

        if self.args.use_depth_text_align:
            self.lenth = args.n_depth_text_tokens
            self.extra_tkns_cfg = {}    # Contains k:v pairs of token name, number of extra tokens with that name.
            self.extra_tkns_learnables = {}   # Contains refs to extra LEARNABLE tokens, if any.  
            self.extra_tkns_lookup = {}   # Key = new token, value = learnable parameter for that token.
            self.extra_tkns_reverse_lookup = {}   # Key = token index, value = learnable parameter for that token.

            # Templates to use
            self.sentence_templates = ["This {object_tkn} appears to be {depth_tkn}"]            
            self.object_tokens = ["patch"]  

            # 256 learned depth tokens. NOT input-dependent: represented as learnable parameters.
            depth_tokens_init = [
                        "very close",
                        "close",
                        "slightly close",
                        "neither close nor distant",
                        "slightly distant",
                        "distant",
                        "very distant"
                    ]
            depth_tokens_init, _ = clip.tokenize(depth_tokens_init)
            with torch.no_grad():
                text_embeddings = self.clip.encode_text(depth_tokens_init)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings / text_embeddings.norm()

            self.depth_tokens = [f"<|depth_{i}|>" for i in range(self.lenth)]
            self.extra_tkns_cfg["depth"] = self.lenth
            if self.args.use_depth_text_embedding_preinit:
                print('===================== preinit depth text embeddings =======================')
                interped_text_embeddings = F.interpolate(text_embedding.transpose(0, 1)[None, None, ...], (512,self.lenth)).squeeze().transpose(0, 1)
                self.extra_tkns_learnables_depth = nn.ParameterList(values=[x[None, :] for x in interped_text_embeddings])
            else:
                self.extra_tkns_learnables_depth = nn.ParameterList(values=[torch.normal(mean=0, std=0.02, size=(1, 512)) for i in range(self.lenth)])
            for i in range(self.lenth):
                self.extra_tkns_lookup[f"<|depth_{i}|>"] = self.extra_tkns_learnables_depth[i]

            # Build templates (word-level). Doesn't handle img_f substitution.
            self.texts = []
            for template in self.sentence_templates:
                for obj in self.object_tokens:
                    for depth in self.depth_tokens:
                        self.texts.append(template.format(object_tkn=obj, depth_tkn=depth))
            self.texts_tokenized, self.tokenizer = clip.tokenize(self.texts, extra_tkns_cfg=self.extra_tkns_cfg)  # tokenize    

    # def save_parameters(self, filename: str) -> None:
    #     """Save the LoRA weights and decoder weights to a .pt file

    #     Args:
    #         filename (str): Filename of the weights
    #     """
    #     w_a, w_b = {}, {}
    #     if self.use_lora:
    #         w_a = {f"w_a_{i:03d}": self.w_a[i].weight for i in range(len(self.w_a))}
    #         w_b = {f"w_b_{i:03d}": self.w_b[i].weight for i in range(len(self.w_a))}

    #     decoder_weights = self.decoder.state_dict()
    #     torch.save({**w_a, **w_b, **decoder_weights}, filename)


    # def load_parameters(self, filename: str) -> None:
    #     """Load the LoRA and decoder weights from a file

    #     Args:
    #         filename (str): File name of the weights
    #     """
    #     state_dict = torch.load(filename)

    #     # Load the LoRA parameters
    #     if self.use_lora:
    #         for i, w_A_linear in enumerate(self.w_a):
    #             saved_key = f"w_a_{i:03d}"
    #             saved_tensor = state_dict[saved_key]
    #             w_A_linear.weight = nn.Parameter(saved_tensor)

    #         for i, w_B_linear in enumerate(self.w_b):
    #             saved_key = f"w_b_{i:03d}"
    #             saved_tensor = state_dict[saved_key]
    #             w_B_linear.weight = nn.Parameter(saved_tensor)

    #     # Load decoder parameters
    #     decoder_head_dict = self.decoder.state_dict()
        
    #     decoder_head_keys = [k for k in decoder_head_dict.keys()]
    #     decoder_state_dict = {k: state_dict[k] for k in decoder_head_keys}

    #     self.decoder.load_state_dict(decoder_state_dict)
    
    def get_depth_language_features(self, img_f_tkns=None, extra_tkns_override=None):
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

    def fuse_dino_clip(self, middle_feats_dino, middle_feats_clip, dino_h=16, dino_w=48):
        assert len(middle_feats_clip) == len(middle_feats_dino)
        fused_feats = []
        for i, (feat_clip, feat_dino) in enumerate(zip(middle_feats_clip, middle_feats_dino)):
            dino_shape = [dino_h, dino_w] if feat_dino is None else feat_dino.shape[-2:]
            # 方法一：通过插值对齐大小，再通过1x1卷积对齐特征维度
            if self.dino_clip_align_method == 'interp_conv1x1':
                aligned_feat_clip = F.interpolate(feat_clip, dino_shape)
                aligned_feat_clip = self.clip_align_conv[i](aligned_feat_clip)
            # 方法二：通过下/上采样卷积尽量对齐大小和通道维度，再通过插值完全对齐大小
            elif self.dino_clip_align_method == 'convs2_interp':
                aligned_feat_clip = self.clip_align_conv[i](feat_clip)
                aligned_feat_clip = F.interpolate(aligned_feat_clip, dino_shape)
            else:
                raise NotImplementedError
            
            if feat_dino is not None:
                fused_feat = torch.concatenate([feat_dino, aligned_feat_clip], dim=1)
            else:
                fused_feat = aligned_feat_clip
            #fused_feat = feat_dino
            fused_feats.append(fused_feat)
        return fused_feats

    def forward(self, image, pose):
        dino_h, dino_w = image.shape[-2] // 14, image.shape[-1] // 14
        if not self.only_clip:
            img_shape_list = [list(image.shape[-2:]) for _ in range(image.shape[0])]
            middle_feats_dino = self.dinov2.get_intermediate_layers(image, [2, 5, 8, 11], return_class_token=True, img_shape_list=img_shape_list, reshape=True)
            middle_feats_dino, middle_dino_cls_tokens = list(zip(*middle_feats_dino))

        # TODO: 怎么把dino和clip(多层)的feature map融合起来，大小和通道不一样
        # if self.use_clip and self.use_dino:
        if self.use_clip:
            if not self.only_clip:
                if not self.use_resnet50_instead_clip:
                    image_features_clip, depth_x, middle_feats_clip = self.clip.encode_image(image,rank=False,csa=False,return_middle=True)
                    middle_feats_clip = list(middle_feats_clip.values())
                else:
                    middle_feats_clip = self.resnet50_forward(image)
                fused_feats = self.fuse_dino_clip(middle_feats_dino, middle_feats_clip)
            else:
                image_features_clip, depth_x, middle_feats_clip = self.clip.encode_image(image,rank=False,csa=False,return_middle=True)
                middle_feats_clip = list(middle_feats_clip.values())
                fused_feats = self.fuse_dino_clip([None]*4, middle_feats_clip)
        else:
            fused_feats = middle_feats_dino

        if self.args.use_depth_text_align:
            img_f_tkns = None
            extra_tkns_override = None
            depth_text_embeddings = self.get_depth_language_features(img_f_tkns, extra_tkns_override)      # 3 dims
        else:
            depth_text_embeddings = None
        depth = dict(depth_text_embeddings=depth_text_embeddings, dino_cls_tokens=middle_dino_cls_tokens[-1] if not self.only_clip else None)
        return None, fused_feats, depth, None