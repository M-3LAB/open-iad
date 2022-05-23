import torch
import torch.nn as nn
from seg_network.modules.block import * 

__all__ = ['DepthSeg']

class DepthSeg(nn.Module):
    def __init__(self, inc, base_width, ouc):
        super(DepthSeg).__init__()

        self.inc = inc
        self.base_width = base_width
        self.ouc = ouc

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
        self.up_bottom = DecUpSegBlock(inc=self.base_width*8, ouc=self.base_width*8)
        self.dec_bottom = DecDownSegBlock(inc=self.base_width*(8+8), ouc=self.base_width*8)

        self.up1 = DecUpSegBlock(inc=self.base_width*8, ouc=self.base_width*4)
        self.dec1 = DecDownSegBlock(inc=self.base_width*(8+4), ouc=self.base_width*4)

        self.up2 = DecUpSegBlock(inc=self.base_width*4, ouc=self.base_width*2)
        self.dec2 = DecDownSegBlock(inc=self.base_width*(4+2), ouc=self.base_width*2)

        self.up3 = DecUpSegBlock(inc=self.base_width*2, ouc=self.base_width)
        self.dec3 = DecDownSegBlock(inc=self.base_width*(2+1), ouc=self.base_width)

        self.up4 = DecUpSegBlock(inc=self.base_width, ouc=self.base_width)
        self.dec4 = DecDownSegBlock(inc=self.base_width*(4+2), ouc=self.base_width)
        
        self.fin_out = nn.Sequential(nn.Conv2d(self.base_width, self.ouc, kernel_size=3, padding=1)) 

    def forward(self, x):
        # Encoder Part

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

        # Decoder Part

        up_b = self.up_bottom(e6)
        cat_b = torch.cat((up_b, e5), dim=1) 
        db = self.dec_bottom(cat_b)

        up_1 = self.up1(db)
        cat_1 = torch.cat((up_1, e4), dim=1)
        d1 = self.dec1(cat_1)

        up_2 = self.up2(d1)
        cat_2 = torch.cat((up_2, e3), dim=1)
        d2 = self.dec2(cat_2)

        up_3 = self.up3(d2)
        cat_3 = torch.cat((up_3, e2), dim=1)
        d3 = self.dec3(cat_3)

        up_4 = self.up4(d3)
        cat_4 = torch.cat((up_4, e1), dim=1)
        d4 = self.dec4(cat_4)

        out = self.fin_out(d4)

        return out