import torch
import torch.nn as nn
from reconstruction_network.modules.block import *

__all__ = ['RGBRecons']

class RGBRecons(nn.Module):

    def __init__(self, inc, base_width):
        super(RGBRecons, self).__init__()
        self.inc = inc
        self.base_width = base_width
        # Encoder Part
        self.enc1 = EncBlock(inc=self.inc, ouc=self.base_width)
        self.enc2 = EncBlock(inc=self.base_width, ouc=self.base_width*2)
        self.enc3 = EncBlock(inc=self.base_width*2, ouc=self.base_width*4)
        self.enc4 = EncBlock(inc=self.base_width*4, ouc=base_width*8)
        self.enc5 = StemBlock(inc=self.base_width*8, ouc=self.base_width*8) 

    def forward(self, x):
        pass