import torch
import torch.nn as nn

__all__ = ['GSMLoss']

class GSMLoss(nn.Module):
    def __init__(self):
        super(GSMLoss).__init__()
    
    def forward(self, x):
        pass