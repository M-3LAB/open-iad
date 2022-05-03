import torch
import torch.nn as nn

__all__ = ['EncBlock', 'DecUpBlock', 'DecDownBlock', 'StemBlock']

"""
DRAEM Stem Block
"""
class StemBlock(nn.Module):
    def __init__(self, inc, ouc, ks=3, padding=1):
        super().__init__()
        self.inc = inc
        self.ouc = ouc
        self.ks = ks
        self.pad = padding

        self.block = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(inplace=True),
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(inplace=True) 
        )
    
    def forward(self, x):
       output = self.block(x) 
       return output

"""
DRAEM Reconstructive Encoder Block
"""
class EncBlock(nn.Module):
    def __init__(self, inc, ouc, ks=3, padding=1):
        super(EncBlock).__init__()
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
            nn.ReLU(inplace=True),
            nn.MaxPooling2d(kernel_size=2)
        ) 
        
    
    def forward(self, x):
        output = self.block(x)
        return output

"""
DRAEM Reconstructive Decoder Upsampling Block
"""

class DecUpBlock(nn.Module):
    def __init__(self, inc, ouc, ks=3, padding=1, up_mode='bilinear', scale_factors=2, align_corners=True):
        super(DecUpBlock).__init__()

        self.inc = inc
        self.ouc = ouc
        self.up_mode = up_mode
        self.scale_factors = scale_factors
        self.align_corners = align_corners
        self.ks = ks
        self.pad = padding

        self.block = nn.Sequential(
            nn.Upsample(scale_factors=self.scale_factors, mode=self.up_mode, align_corners=self.align_corners),
            nn.Conv2d(self.inc, self.ouc, kernel_size=self.ks, padding=self.pad),
            nn.BatchNorm2d(self.inc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        output = self.block(x) 
        return output

"""
DRAEM Reconstructive Decoder Downsampling Block
"""

class DecDownBlock(nn.Module):
    def __init__(self, inc, ouc, ks=3, padding=1):
        super(DecDownBlock).__init__()
        self.inc = inc
        self.ks = ks
        self.pad = padding
        self.ouc = ouc

        self.block = nn.Sequential(
            nn.Conv2d(self.inc, self.inc, kernel_size=self.ks, padding=self.pad),
            nn.BatchNorm2d(self.inc),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inc, self.ouc, kernel_size=self.ks, padding=self.pad),
            nn.BatchNorm2d(self.ouc),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        output = self.block(x)
        return output
