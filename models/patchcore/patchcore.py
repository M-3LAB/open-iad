import torch
import torch.nn as nn
import numpy as np

__all__ = ['PatchCore']

class PatchCore(torch.nn.Module):
    def __init__(self, backbone, device):
        super(PatchCore).__init__()
    
    def forward(self, x):
        pass