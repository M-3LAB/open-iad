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
        self.activation = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(input_channles=self.inc, out_channels=self.ouc, kernel_size=self.ks, bias=self.bias),
            nn.BatchNorm1d(num_features=self.ouc),
            self.activation)


    def forward(self, x):
       output = self.net(x) 
       return output