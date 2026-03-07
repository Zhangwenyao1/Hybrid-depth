# Ref: https://github.com/rawmarshmellows/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from . import clip


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super(Bridge, self).__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlock(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super(UpBlock, self).__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class MSAdapter(nn.Module):
    def __init__(self, in_dim=2048, n_class=1, use_seg=False):
        super(MSAdapter, self).__init__()
        self.n_class = n_class
        self.bridge = Bridge(in_dim, in_dim)

        up_blocks = []
        up_blocks.append(UpBlock(in_dim, in_dim // 2))
        up_blocks.append(UpBlock(in_dim // 2, in_dim // 4))
        up_blocks.append(UpBlock(in_dim // 4, in_dim // 8))

        up_blocks.append(UpBlock(in_channels=(in_dim // 16) + (in_dim // 32), out_channels=(in_dim // 16), \
            up_conv_in_channels=(in_dim // 8), up_conv_out_channels=(in_dim // 16)))

        if use_seg:
            up_blocks.append(UpBlock(in_channels=(in_dim // 32) + 6, out_channels=(in_dim // 32), \
                up_conv_in_channels=(in_dim // 16), up_conv_out_channels=(in_dim // 32)))
        else:
            up_blocks.append(UpBlock(in_channels=(in_dim // 32) + 3, out_channels=(in_dim // 32), \
                up_conv_in_channels=(in_dim // 16), up_conv_out_channels=(in_dim // 32)))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(in_dim // 32, n_class, kernel_size=1, stride=1)

    def forward(self, features):
        x = self.bridge(features['down5'])
        #print(f'bridge: {x.shape}')

        for i, block in enumerate(self.up_blocks, 1):
            key = f"down{5-i}"
            x = block(x, features[key])
            #print(f'up {i}: {x.shape}')

        x = self.out(x)
        #print(f'out: {x.shape}')

        return x


class DepthConvAdapter(nn.Module):
    def __init__(self, **cfg):
        super(DepthConvAdapter, self).__init__()
        self.depth_classes = cfg['depth_classes']
        self.max_depth = cfg['max_depth']
        self.bin_list = cfg['bin_list']
        self.n_bin = len(self.bin_list)
        self.temperature = cfg['temperature']
        self.obj_classes = ['object']
        self.depth_templates = ['This {} is {}']
        self.use_seg = cfg['use_seg']

        print("Loading CLIP (backbone:" + cfg['name'] + ")")
        self.clip, _ = clip.load(cfg['name']) # load pretrained clip encoder
        self.dtype = self.clip.dtype
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.clip.named_parameters():
            param.requires_grad_(False)
        self.clip.eval()

        # u-net like multi-scale adapter
        self.alpha = cfg['clip_weight'] # image adapter weight
        self.image_adapter = MSAdapter(cfg['image_adapter']['input_size'], use_seg=self.use_seg).to(self.dtype)
        self.seg_encoder = Bridge(3, 3).to(self.dtype)
        self.image_adapter.train()
        self.seg_encoder.train()

        # double check
        enabled = set()
        n_param = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                n_param += param.numel()
        print(f"Parameters to be updated: {enabled} (# of params: {n_param})")

    def encode_text(self):
        with torch.no_grad():
            text_features = []
            for depth in self.depth_classes:
                for obj in self.obj_classes:
                    texts = [template.format(obj, depth) for template in self.depth_templates]
                    texts = clip.tokenize(texts).cuda()  # tokenize
                    class_embeddings = self.clip.encode_text(texts)  # embed with text encoder
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    text_features.append(class_embedding)
            text_features = torch.stack(text_features, dim=1)

        return text_features

    def encode_image(self, x, seg=None):
        H, W = x.shape[2:]
        with torch.no_grad():
            image_features, down_features = self.clip.encode_image(x.type(self.dtype), return_feat=True)
            image_features = image_features.permute(1, 0, 2)  # B, HW, C

        if self.use_seg:
            down_features['down0'] = torch.cat((down_features['down0'], self.seg_encoder(seg.type(self.dtype))), dim=1)
        x = self.image_adapter(down_features)

        return image_features, x

    def forward(self, x, seg=None):
        H, W = x.shape[2:]
        text_f = self.encode_text()
        img_f, delta = self.encode_image(x, seg)

        # scaling images to match the dimension of text embeddings
        scale = float(text_f.size(0)) / img_f.size(-1)
        img_f = F.interpolate(img_f, scale_factor=scale)
        text_f = F.normalize(text_f, dim=0)
        img_f = F.normalize(img_f, dim=-1)

        # compute similarity score between text and image embeddings
        depth_logits = img_f @ text_f  # B, HW, K
        H, W = H // 16, W // 16
        if H * W > depth_logits.size(1):
            H, W = H // 2, W // 2
        depth_logits = depth_logits.permute(0, 2, 1).view(-1, self.n_bin, H, W)  # B, K, H, W
        depth_logits /= self.temperature

        depth = F.softmax(depth_logits, dim=1)
        bin_tensor = torch.tensor(self.bin_list).to(depth.device)
        depth = depth * bin_tensor.view(1, self.n_bin).unsqueeze(-1).unsqueeze(-1)
        depth = depth.sum(1, keepdim=True)
        depth = depth.clamp(0, self.max_depth)
        depth = F.interpolate(depth, size=delta.shape[-2:], mode='bilinear', align_corners=True)
        if self.alpha > 0:
            delta = (F.sigmoid(delta) - 0.5) * 2
        depth = self.alpha * depth + delta
        return depth, depth_logits


def depth_convadapter(**kwargs):
    model = DepthConvAdapter(**kwargs)
    return model


