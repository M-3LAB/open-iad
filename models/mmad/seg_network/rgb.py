import torch
import torch.nn as nn
from seg_network.modules.block import * 

__all__=['RGBSeg']

class RGBSeg(nn.Module):
    def __init__(self, inc, base_width):
        super(RGBSeg, self).__init__()

        self.inc = inc
        self.base_width = base_width

        # Encoder Part
        self.enc1 = EncSegBlock(inc=self.inc, ouc=self.base_width)
        self.mp1 = nn.Sequential(nn.MaxPool2d(kernel_size=2))

        self.enc2 = EncSegBlock(inc=self.base_width*2, ouc=self.base_width*4)
        self.mp2 = nn.Sequential(nn.MaxPool2d(kernel_size=2))

        self.enc3 = EncSegBlock(inc=self.base_width*4, ouc=self.base_width*8)
        self.mp3 = nn.Sequential(nn.MaxPool2d(kernel_size=2))

        self.enc4 = EncSegBlock(inc=self.base_width*4, ouc=self.base_width*8)
        self.mp4 = nn.Sequnetial(nn.MaxPool2d(kernel_size=2))

        self.enc5 = EncSegBlock(inc=self.base_width*8, ouc=self.base_width*8)
        self.mp4 = nn.Sequnetial(nn.MaxPool2d(kernel_size=2))

        self.enc6 = EncSegBlock(inc=self.base_width*8, ouc=self.base_width*8)
        

    def forward(self, x):
        pass