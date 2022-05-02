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

    def forward(self, x):
        pass