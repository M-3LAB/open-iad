import torch
import torch.nn as nn
from reconstruction_network.modules.block import *

__all__ = ['DepthRecons']

class DepthRecons(nn.Module):
    def __init__(self, inc, base_width, fin_ouc):
        super(DepthRecons).__init__()
        self.inc = inc
        self.base_width = base_width
        # Final output channel
        self.fin_ouc = fin_ouc

    def forward(self, x):
        pass