import torch
import torch.nn as nn

__all__ = ['EncSegBlock', 'DecSegBlock']

class EncSegBlock(nn.Module):

    def __init__(self, inc, ouc, ks=3, padding=1):
        super(EncSegBlock).__init__()

        self.inc = inc
        self.ouc = ouc 
        self.ks = ks
        self.pad = padding

        self.block = nn.Sequential(
            nn.Conv2d(self.inc, self.ouc, kernel_size=self.ks, padding=self.pad),
            nn.BatchNorm2d(self.ouc),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.ouc, self.ouc, kernel_size=self.ks, padding=self.pad),
            nn.BatchNorm2d(self.ouc),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        output = self.block(x)
        return output


class DecSegBlock(nn.Module):

    def __init__(self, inc, ouc):
        super(DecSegBlock).__init__()

        self.inc = inc
        self.ouc = ouc
    
    def forward(self, x):
        pass
