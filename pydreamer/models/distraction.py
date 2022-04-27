from typing import Optional, Union
import torch
import torch.nn as nn
import torch.distributions as D

from .functions import *
from .common import *


class DistractionEncoder(nn.Module):

    def __init__(self, in_channels=3, cnn_depth=32, activation=nn.ELU):
        super().__init__()
        self.out_dim = cnn_depth * 32
        kernels = (4, 4, 4, 4)
        stride = 2
        d = cnn_depth
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, d, kernels[0], stride),
            activation(),
            nn.Conv2d(d, d * 2, kernels[1], stride),
            activation(),
            nn.Conv2d(d * 2, d * 4, kernels[2], stride),
            activation(),
            nn.Conv2d(d * 4, d * 8, kernels[3], stride),
            activation(),
            nn.Flatten()
        )

    def forward(self, x, p):
        x, bd = flatten_batch(x - p, 3)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y


class DiffEncoder:
    def __init__(self):
        self.prev_obs = None

    def calc_diff(self, obs):
        pass


