import torch
import torch.nn as nn

__all__ = ['EncSegBlock', 'DecUpSegBlock', 'DecDownSegBlock']

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


class DecDownSegBlock(nn.Module):

    def __init__(self, inc, ouc, ks=3, padding=1):
        super(DecDownSegBlock).__init__()

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

class DecUpSegBlock(nn.Module):

    def __init__(self, inc, ouc, scale_factor=2, mode='bilinear', align_corners=True):
        super(DecUpSegBlock).__init__()

        self.inc = inc
        self.ouc = ouc
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=self.scale_factor),
            nn.Conv2d(self.inc, self.ouc, kernel_size=self.ks),
            nn.BatchNorm2d(self.ouc),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        output = self.block(x)
        return output
