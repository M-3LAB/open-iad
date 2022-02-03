import torch
import torch.nn as nn

__all__ = ['PatchCore']

class PatchCore(torch.nn.Module):
    def __init__(self):
        super(PatchCore).__init__()
    
    def fit(self, x):
        pass