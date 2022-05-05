import torch
import torch.nn as nn

__all__ = ['EncSegBlock']

class EncSegBlock(nn.Module):

    def __init__(self, inc, base_width):
        super().__init__()

        self.inc = inc
        self.base_width = base_width
    
    def forward(self, x):
        pass
