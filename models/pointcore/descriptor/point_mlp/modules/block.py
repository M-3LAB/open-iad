import torch
import torch.nn as nn
from point_mlp.modules.activation import get_activation

__all__ = ['MLP1D']

class MLP1D(nn.Module):
    def __init__(self, inc, ouc, ks=1, bias=True, activation='relu'):
        super(MLP1D).__init__()
        self.inc = inc
        self.ouc = ouc
        self.ks = ks
        self.bias = bias
        self.activation = activation

    def forward(self, x):
        pass