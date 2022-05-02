import torch
import torch.nn as nn

__all__ = ['EncBlock', 'DecBlock']

"""
DRAEM Reconstructive Encoder Block
"""
class EncBlock(nn.Module):
    def __init__(self, inc, expansion_ratio, ks, padding=1):
        super(EncBlock).__init__()
        self.inc = inc
        self.exp = expansion_ratio
        self.ks = ks
        self.pad = padding
        self.block = nn.Sequential(
            nn.Conv2d(self.inc, self.inc*self.exp, kernel_size=self.ks, padding=self.pad),
            nn.BatchNorm2d(self.inc*self.exp),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inc*self.exp, self.inc*self.exp, kernel_size=self.ks, padding=self.pad),
            nn.BatchNorm2d(self.inc*self.exp),
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
    def __init__(self, inc, up_mode='bilinear', scale_factors=2, align_corners=True):
        super(DecUpBlock).__init__()

    def forward(self, x):
        pass
