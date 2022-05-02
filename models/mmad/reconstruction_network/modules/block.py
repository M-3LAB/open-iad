from tkinter import W
import torch
import torch.nn as nn

__all__ = ['EncBlock']

"""
DRAEM Reconstructive Encoder Block
"""
class EncBlock(nn.Module):
    def __init__(self, inc, expansion_ratio):
        super(EncBlock).__init__()
        self.inc = inc
        self.expansion_ratio = expansion_ratio
        self.block = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.MaxPooling()
        ) 
        
    
    def forward(self, x):
        output = self.block(x)
        return output
