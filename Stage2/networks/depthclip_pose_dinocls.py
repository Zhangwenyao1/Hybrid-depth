# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class depthclippose_dinocls(nn.Module):
    def __init__(self, in_channels):
        super(depthclippose_dinocls, self).__init__()

        self.net = nn.Linear(in_channels, 6)

    def forward(self, input_features):
        out = self.net(input_features)
        out = 0.01 * out.view(-1, 1, 1, 6)
        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation