import torch
import torch.nn as nn
from seg_network.modules.block import * 

__all__ = ['DepthSeg']

class DepthSeg(nn.Module):
    def __init__(self, inc, base_width):
        super(DepthSeg).__init__()

        self.inc = inc
        self.base_width = base_width

    def forward(self, x):
        pass