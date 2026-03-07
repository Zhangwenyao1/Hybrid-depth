# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from layers import BackprojectDepth, Project3D
from CLIP import clip

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        #loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        print('resnet checkpoint loaded')
        loaded = torch.load('/code/CFMDE-main/resnet18-f37072fd.pth')
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class DINOCLIPEncoderMatching(nn.Module):
    """Resnet encoder adapted to include a cost volume after the 2nd block.

    Setting adaptive_bins=True will recompute the depth bins used for matching upon each
    forward pass - this is required for training from monocular video as there is an unknown scale.
    """

    def __init__(self, args, input_height, input_width,
                 min_depth_bin=0.1, max_depth_bin=20.0, num_depth_bins=96,
                 adaptive_bins=False, depth_binning='linear'):

        super(DINOCLIPEncoderMatching, self).__init__()
        self.args = args
        self.adaptive_bins = adaptive_bins
        self.depth_binning = depth_binning
        self.set_missing_to_max = True
        self.backbone_model = args.backbone_model

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.num_depth_bins = num_depth_bins
        # we build the cost volume at 1/4 resolution
        self.matching_height_clip, self.matching_width_clip = input_height // 4, input_width // 4
        self.matching_height_dino, self.matching_width_dino = input_height // 14, input_width // 14

        self.is_cuda = False
        self.warp_depths = None
        self.depth_bins = None

        self.dinov2:nn.Module = torch.hub.load('/code/CFMDE-main/dinov2', 'dinov2_vitb14', source='local', verbose=True, pretrained=False)
        self.dinov2.load_state_dict(torch.load('/model/ericliu/DINOv2/dinov2_vitb14_pretrain.pth'))
        self.clip, self.clip_preprocess = clip.load('RN50', device="cpu", download_root='/code/CFMDE-main/checkpoints')

        #self.layer0_dino = self.dinov2.patch_embed
        self.layer0_dino = self.dinov2.prepare_tokens_with_masks
        self.layer1_dino = nn.Sequential(*self.dinov2.blocks[:3])
        self.layer2_dino = nn.Sequential(*self.dinov2.blocks[3:6])
        self.layer3_dino = nn.Sequential(*self.dinov2.blocks[6:9])
        self.layer4_dino = nn.Sequential(*self.dinov2.blocks[9:12])
        self.dino_norm = self.dinov2.norm

        self.layer0_clip = nn.Sequential(self.clip.visual.conv1, self.clip.visual.bn1, self.clip.visual.relu1, self.clip.visual.conv2, self.clip.visual.bn2, self.clip.visual.relu2, self.clip.visual.conv3, self.clip.visual.bn3, self.clip.visual.relu3)
        self.layer1_clip = nn.Sequential(self.clip.visual.avgpool, self.clip.visual.layer1)
        self.layer2_clip = self.clip.visual.layer2
        self.layer3_clip = self.clip.visual.layer3
        self.layer4_clip = self.clip.visual.layer4
        self.dinov2.to('cuda')
        #del self.dinov2
        #del self.clip

        # self.layer0 = nn.Sequential(encoder.conv1,  encoder.bn1, encoder.relu)
        # self.layer1 = nn.Sequential(encoder.maxpool,  encoder.layer1)
        # self.layer2 = encoder.layer2
        # self.layer3 = encoder.layer3
        # self.layer4 = encoder.layer4

        self.backprojector_clip = BackprojectDepth(batch_size=self.num_depth_bins,
                                              height=self.matching_height_clip,
                                              width=self.matching_width_clip)
        self.projector_clip = Project3D(batch_size=self.num_depth_bins,
                                   height=self.matching_height_clip,
                                   width=self.matching_width_clip)
        
        self.backprojector_dino = BackprojectDepth(batch_size=self.num_depth_bins,
                                              height=self.matching_height_dino,
                                              width=self.matching_width_dino)
        self.projector_dino = Project3D(batch_size=self.num_depth_bins,
                                   height=self.matching_height_dino,
                                   width=self.matching_width_dino)

        self.warp_depths = dict()
        self.compute_depth_bins(min_depth_bin, max_depth_bin, 0)
        self.compute_depth_bins(min_depth_bin, max_depth_bin, 1)

        self.prematching_conv = nn.Sequential(nn.Conv2d(64, out_channels=16,
                                                        kernel_size=1, stride=1, padding=0),
                                              nn.ReLU(inplace=True)
                                              )

        self.reduce_conv = nn.ModuleList([
                                nn.Sequential(nn.Conv2d(256 + self.num_depth_bins,
                                                   out_channels=256,
                                                   kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(inplace=True)
                                         ),
                                nn.Sequential(nn.Conv2d(768 + self.num_depth_bins,
                                                   out_channels=768,
                                                   kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(inplace=True)
                                         ),
                                
        ])
        clip_aligned_dim = 256
        # fuse clip and dino
        self.dino_clip_align_method = 'interp_conv1x1'#self.args.dino_clip_align_method
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
    
    def compute_depth_bins(self, min_depth_bin, max_depth_bin, flag):
        """Compute the depths bins used to build the cost volume. Bins will depend upon
        self.depth_binning, to either be linear in depth (linear) or linear in inverse depth
        (inverse)"""

        if self.depth_binning == 'inverse':
            self.depth_bins = 1 / np.linspace(1 / max_depth_bin,
                                              1 / min_depth_bin,
                                              self.num_depth_bins)[::-1]  # maintain depth order

        elif self.depth_binning == 'linear':
            self.depth_bins = np.linspace(min_depth_bin, max_depth_bin, self.num_depth_bins)
        else:
            raise NotImplementedError
        self.depth_bins = torch.from_numpy(self.depth_bins).float()

        warp_depths = []
        for depth in self.depth_bins:
            if flag==0:
                depth = torch.ones((1, self.matching_height_clip, self.matching_width_clip)) * depth
            else:
                depth = torch.ones((1, self.matching_height_dino, self.matching_width_dino)) * depth
            warp_depths.append(depth)
        warp_depths = torch.stack(warp_depths, 0).float()
        if self.is_cuda:
            warp_depths = warp_depths.cuda()
        self.warp_depths[flag] = warp_depths

    def match_features(self, current_feats, lookup_feats, relative_poses, K, invK, flag):
        """Compute a cost volume based on L1 difference between current_feats and lookup_feats.

        We backwards warp the lookup_feats into the current frame using the estimated relative
        pose, known intrinsics and using hypothesised depths self.warp_depths (which are either
        linear in depth or linear in inverse depth).

        If relative_pose == 0 then this indicates that the lookup frame is missing (i.e. we are
        at the start of a sequence), and so we skip it"""

        batch_cost_volume = []  # store all cost volumes of the batch
        cost_volume_masks = []  # store locations of '0's in cost volume for confidence
        warp_depths = self.warp_depths[flag]
        if flag == 0:
            backprojector = self.backprojector_clip
            projector = self.projector_clip
            matching_height = self.matching_height_clip
            matching_width = self.matching_width_clip
        else:
            backprojector = self.backprojector_dino
            projector = self.projector_dino
            matching_height = self.matching_height_dino
            matching_width = self.matching_width_dino
        for batch_idx in range(len(current_feats)):

            volume_shape = (self.num_depth_bins, matching_height, matching_width)
            cost_volume = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)
            counts = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)

            # select an item from batch of ref feats
            _lookup_feats = lookup_feats[batch_idx:batch_idx + 1]
            _lookup_poses = relative_poses[batch_idx:batch_idx + 1]

            _K = K[batch_idx:batch_idx + 1]
            _invK = invK[batch_idx:batch_idx + 1]
            world_points = backprojector(warp_depths, _invK)

            # loop through ref images adding to the current cost volume
            for lookup_idx in range(_lookup_feats.shape[1]):
                lookup_feat = _lookup_feats[:, lookup_idx]  # 1 x C x H x W
                lookup_pose = _lookup_poses[:, lookup_idx]

                # ignore missing images
                if lookup_pose.sum() == 0:
                    continue

                lookup_feat = lookup_feat.repeat([self.num_depth_bins, 1, 1, 1])
                pix_locs = projector(world_points, _K, lookup_pose)
                warped = F.grid_sample(lookup_feat, pix_locs, padding_mode='zeros', mode='bilinear',
                                       align_corners=True)

                # mask values landing outside the image (and near the border)
                # we want to ignore edge pixels of the lookup images and the current image
                # because of zero padding in ResNet
                # Masking of ref image border
                x_vals = (pix_locs[..., 0].detach() / 2 + 0.5) * (
                    matching_width - 1)  # convert from (-1, 1) to pixel values
                y_vals = (pix_locs[..., 1].detach() / 2 + 0.5) * (matching_height - 1)

                edge_mask = (x_vals >= 2.0) * (x_vals <= matching_width - 2) * \
                            (y_vals >= 2.0) * (y_vals <= matching_height - 2)
                edge_mask = edge_mask.float()

                # masking of current image
                current_mask = torch.zeros_like(edge_mask)
                current_mask[:, 2:-2, 2:-2] = 1.0
                edge_mask = edge_mask * current_mask

                diffs = torch.abs(warped - current_feats[batch_idx:batch_idx + 1]).mean(
                    1) * edge_mask

                # integrate into cost volume
                cost_volume = cost_volume + diffs
                counts = counts + (diffs > 0).float()
            # average over lookup images
            cost_volume = cost_volume / (counts + 1e-7)

            # if some missing values for a pixel location (i.e. some depths landed outside) then
            # set to max of existing values
            missing_val_mask = (cost_volume == 0).float()
            if self.set_missing_to_max:
                cost_volume = cost_volume * (1 - missing_val_mask) + \
                    cost_volume.max(0)[0].unsqueeze(0) * missing_val_mask
            batch_cost_volume.append(cost_volume)
            cost_volume_masks.append(missing_val_mask)

        batch_cost_volume = torch.stack(batch_cost_volume, 0)
        cost_volume_masks = torch.stack(cost_volume_masks, 0)

        return batch_cost_volume, cost_volume_masks

    def feature_extraction_clip(self, image, return_all_feats=False):
        """ Run feature extraction on an image - first 2 blocks of ResNet"""

        #image = (image - 0.45) / 0.225  # imagenet normalisation
        image = (image - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].cuda()) / torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].cuda()
        feats_0 = self.layer0_clip(image)
        feats_1 = self.layer1_clip(feats_0)

        if return_all_feats:
            return [feats_0, feats_1]
        else:
            return feats_1
        
    def feature_extraction_dino(self, image, return_all_feats=False):
        """ Run feature extraction on an image - first 2 blocks of ResNet"""

        #image = (image - 0.45) / 0.225  # imagenet normalisation
        image = (image - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].cuda()) / torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].cuda()
        feats_0 = self.layer0_dino(image)
        feats_1 = self.layer1_dino(feats_0)
        B = feats_1.shape[0]
        cls_token_0 = feats_0[:, 0, :]
        cls_token_1 = feats_1[:, 0, :]
        feats_0 = feats_0[:, 1:, :].reshape(B, 16, 48, -1).permute(0, 3, 1, 2).contiguous()
        feats_1 = feats_1[:, 1:, :].reshape(B, 16, 48, -1).permute(0, 3, 1, 2).contiguous()
        if return_all_feats:
            return [feats_0, feats_1], [cls_token_0, cls_token_1]
        else:
            return feats_1, cls_token_1

    def indices_to_disparity(self, indices):
        """Convert cost volume indices to 1/depth for visualisation"""

        batch, height, width = indices.shape
        depth = self.depth_bins[indices.reshape(-1).cpu()]
        disp = 1 / depth.reshape((batch, height, width))
        return disp

    def compute_confidence_mask(self, cost_volume, num_bins_threshold=None):
        """ Returns a 'confidence' mask based on how many times a depth bin was observed"""

        if num_bins_threshold is None:
            num_bins_threshold = self.num_depth_bins
        confidence_mask = ((cost_volume > 0).sum(1) == num_bins_threshold).float()

        return confidence_mask

    def forward(self, current_image, lookup_images, poses, K, invK,
                min_depth_bin=None, max_depth_bin=None
                ):

        # feature extraction
        model_to_use = []
        self.features_clip = None
        self.features_dino = None
        if 'clip' in self.backbone_model:
            self.features_clip = self.feature_extraction_clip(current_image, return_all_feats=True)
            current_feats_clip = self.features_clip[-1]
            model_to_use.append(0)
        if 'dino' in self.backbone_model:
            if self.args.no_cost_volume_in_dino:
                img_shape_list = [list(current_image.shape[-2:]) for _ in range(current_image.shape[0])]
                middle_feats_dino = self.dinov2.get_intermediate_layers(current_image, [2, 5, 8, 11], return_class_token=True, img_shape_list=img_shape_list, reshape=True)
                self.features_dino, _ = list(zip(*middle_feats_dino))
                self.features_dino = [None, *self.features_dino]
            else:
                self.features_dino, cls_token = self.feature_extraction_dino(current_image, return_all_feats=True)
                current_feats_dino = self.features_dino[-1]
                model_to_use.append(1)
            
        # feature extraction on lookup images - disable gradients to save memory
        for i in model_to_use:
            with torch.no_grad():
                if self.adaptive_bins:
                    self.compute_depth_bins(min_depth_bin, max_depth_bin, i)

                batch_size, num_frames, chns, height, width = lookup_images.shape
                lookup_images_ = lookup_images.reshape(batch_size * num_frames, chns, height, width)
                if i == 0:
                    lookup_feats = self.feature_extraction_clip(lookup_images_,
                                                    return_all_feats=False)
                else:
                    lookup_feats, _ = self.feature_extraction_dino(lookup_images_,
                                                    return_all_feats=False)
                    
                _, chns, height, width = lookup_feats.shape
                lookup_feats = lookup_feats.reshape(batch_size, num_frames, chns, height, width)

                # warp features to find cost volume
                current_feats = current_feats_clip if i==0 else current_feats_dino
                cost_volume, missing_mask = \
                    self.match_features(current_feats, lookup_feats, poses, K, invK, i)
                confidence_mask = self.compute_confidence_mask(cost_volume.detach() *
                                                            (1 - missing_mask.detach()))

            # for visualisation - ignore 0s in cost volume for minimum
            viz_cost_vol = cost_volume.clone().detach()
            viz_cost_vol[viz_cost_vol == 0] = 100
            mins, argmin = torch.min(viz_cost_vol, 1)
            lowest_cost = self.indices_to_disparity(argmin)

            # mask the cost volume based on the confidence
            cost_volume *= confidence_mask.unsqueeze(1)
            
            if i == 0:
                post_matching_feats = self.reduce_conv[i](torch.cat([self.features_clip[-1], cost_volume], 1))
                self.features_clip.append(self.layer2_clip(post_matching_feats))
                self.features_clip.append(self.layer3_clip(self.features_clip[-1]))
                self.features_clip.append(self.layer4_clip(self.features_clip[-1]))
            else:
                post_matching_feats = self.reduce_conv[i](torch.cat([self.features_dino[-1], cost_volume], 1))
                # (B, 768, H*W)
                post_matching_feats = post_matching_feats.reshape(*post_matching_feats.shape[:2],-1)
                # (B, H*W, 768)
                post_matching_feats = post_matching_feats.permute(0, 2, 1)
                post_matching_feats = torch.cat([cls_token[-1].unsqueeze(1), post_matching_feats], dim=1)
                
                self.features_dino.append(self.layer2_dino(post_matching_feats))
                self.features_dino.append(self.layer3_dino(self.features_dino[-1]))
                self.features_dino.append(self.layer4_dino(self.features_dino[-1]))
                for feat_idx in range(-4, 0):
                    if feat_idx==-4:
                        temp = self.features_dino[feat_idx]
                        temp = temp.reshape(*temp.shape[:2],-1).permute(0, 2, 1)
                        temp = torch.cat([cls_token[-1].unsqueeze(1), temp], dim=1)
                        self.features_dino[feat_idx] = self.dino_norm(temp)
                    else:
                        self.features_dino[feat_idx] = self.dino_norm(self.features_dino[feat_idx])
                for feat_idx in range(-4, 0):
                    B = self.features_dino[feat_idx].shape[0]
                    self.features_dino[feat_idx] = self.features_dino[feat_idx][:, 1:, :].reshape(B, 16, 48, -1).permute(0, 3, 1, 2).contiguous()

        if self.features_clip is not None and self.features_dino is not None:
            self.features = self.fuse_dino_clip(self.features_dino[1:], self.features_clip[1:])
        elif self.features_clip is not None and self.features_dino is None:
            if self.args.use_resnet_decoder:
                self.features = self.features_clip
            else:
                self.features = self.fuse_dino_clip([None]*4, self.features_clip[1:])
        else:
            self.features = self.features_dino[1:]


        if self.args.use_depth_text_align:
            img_f_tkns = None
            extra_tkns_override = None
            depth_text_embeddings = self.get_depth_language_features(img_f_tkns, extra_tkns_override)      # 3 dims
        else:
            depth_text_embeddings = None

        if depth_text_embeddings is not None:
            return {'depth_text_embeddings': depth_text_embeddings, 'features': self.features}, lowest_cost, confidence_mask
        else:
            return self.features, lowest_cost, confidence_mask

    def cuda(self):
        super().cuda()
        self.backprojector_clip.cuda()
        self.backprojector_dino.cuda()
        self.projector_clip.cuda()
        self.projector_dino.cuda()
        self.is_cuda = True
        if self.warp_depths is not None:
            self.warp_depths[0] = self.warp_depths[0].cuda()
            self.warp_depths[1] = self.warp_depths[1].cuda()

    def cpu(self):
        super().cpu()
        self.backprojector_clip.cpu()
        self.backprojector_dino.cpu()
        self.projector_clip.cpu()
        self.projector_dino.cpu()
        self.is_cuda = False
        if self.warp_depths is not None:
            self.warp_depths[0] = self.warp_depths[0].cpu()
            self.warp_depths[1] = self.warp_depths[1].cpu()

    def to(self, device):
        if str(device) == 'cpu':
            self.cpu()
        elif str(device) == 'cuda':
            self.cuda()
        else:
            raise NotImplementedError
        
class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1, **kwargs):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            if num_layers == 18:
                print('resnet checkpoint loaded')
                loaded = torch.load('/code/CFMDE-main/resnet18-f37072fd.pth')
                self.encoder = resnets[num_layers]()
                self.encoder.load_state_dict(loaded)
            else:
                self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
