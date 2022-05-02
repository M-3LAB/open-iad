import torch
import torch.nn as nn
from reconstruction_network.modules.block import *

__all__ = ['RGBRecons']

class RGBRecons(nn.Module):

    def __init__(self, inc, base_width, expansion_ratio=2):
        super(RGBRecons, self).__init__()
        self.inc = inc
        self.base_width = base_width
        self.exp = expansion_ratio
        self.stem = StemBlock(inc=self.inc, ouc=self.base_width)
        # Encoder Part
        self.enc1 = EncBlock(inc=self.base_width, expansion_ratio=self.exp)
        self.enc2 = EncBlock(inc=self.base_width*2, expansion_ratio=self.exp)
        self.enc3 = EncBlock(inc=self.base_width*4, expansion_ratio=self.exp)
        self.enc4 = EncBlock(inc=self.base_width*8, expansion_ratio=self.exp) 

    def forward(self, x):
        pass