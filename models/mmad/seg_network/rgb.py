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

        #Decoder Part
        self.up1 = DecUpSegBlock(inc=self.base_width*8, ouc=self.base_width*4)
        self.dec1 = DecDownSegBlock(inc=self.base_width*(8+4), ouc=self.base_width*4)

        self.up2 = DecUpSegBlock(inc=self.base_width*4, ouc=self.base_width*2)
        self.dec2 = DecDownSegBlock(inc=self.base_width*(4+2), ouc=self.base_width*2)

        self.up3 = DecUpSegBlock(inc=self.base_width*2, ouc=self.base_width)
        self.dec3 = DecDownSegBlock(inc=self.base_width*(2+1), ouc=self.base_width)

        self.up4 = DecUpSegBlock(inc=self.base_width, ouc=self.base_width)
        self.dec4 = DecDownSegBlock(inc=self.base_width*(4+2), ouc=self.base_width)
        
        self.fin_out = nn.Sequential(nn.Conv2d()) 
        

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.mp1(e1)

        e2 = self.enc2(p1)
        p2 = self.mp2(e2)

        e3 = self.enc3(p2)
        p3 = self.mp3(e3)

        e4 = self.enc4(p3)
        p4 = self.mp4(e4)

        e5 = self.enc5(p4)
        p5 = self.mp5(e5)

        e6 = self.enc6(p5)

