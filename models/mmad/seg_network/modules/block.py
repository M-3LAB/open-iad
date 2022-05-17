import torch
import torch.nn as nn

__all__ = ['EncSegBlock', 'DecSegBlock']

class EncSegBlock(nn.Module):

    def __init__(self, inc, ouc):
        super(EncSegBlock).__init__()

        self.inc = inc
        self.ouc = ouc 
    
    def forward(self, x):
        pass

class DecSegBlock(nn.Module):

    def __init__(self, inc, ouc):
        super(DecSegBlock).__init__()

        self.inc = inc
        self.ouc = ouc
    
    def forward(self, x):
        pass
