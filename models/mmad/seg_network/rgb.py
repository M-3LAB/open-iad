import torch
import torch.nn as nn
from seg_network.modules.block import * 

__all__=['RGBSeg']

class RGBSeg(nn.Module):
    def __init__(self, inc, base_width):
        super(RGBSeg, self).__init__()
        self.inc = inc
        self.base_width = base_width
        # Encoder Part
        

    def forward(self, x):
        pass