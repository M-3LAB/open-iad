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

class MLPRes1D(nn.Module):

    def __init__(self, inc, ks=1, bias=True, groups=1, res_expansion=1.0, activation='relu'):
        super(MLPRes1D).__init__()
        self.inc = inc
        self.ks = ks
        self.bias = bias
        self.activation = get_activation(activation)
        self.res_expansion = res_expansion
        self.groups = groups

        # main branch first part -- the part before group convolution
        self.net1 = nn.Sequential(
            nn.Conv1d(input_channles=self.inc, out_channels=int(self.inc * self.res_expansion), 
                      kernel_size=self.ks, groups=self.groups, bias=self.bias),
            nn.BatchNorm1d(num_features=int(self.inc * self.res_expansion)),
            self.activation)
        
        
        
    



    def forward(self, x):
       output = self.net(x) 
       return output