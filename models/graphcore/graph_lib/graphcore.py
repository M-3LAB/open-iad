import torch
import torch.nn as nn


__all__ = ['GraphCore']

class GraphCore(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def forward(self, x):
        pass