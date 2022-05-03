import torch
import torch.nn as nn
from reconstruction_network.modules.block import *

__all__ = ['RGBRecons']

class RGBRecons(nn.Module):

    def __init__(self, inc, base_width, fin_ouc):
        super(RGBRecons, self).__init__()
        self.inc = inc
        self.base_width = base_width
        self.fin_ouc = fin_ouc

        # Encoder Part
        self.enc1 = EncBlock(inc=self.inc, ouc=self.base_width)
        self.enc2 = EncBlock(inc=self.base_width, ouc=self.base_width*2)
        self.enc3 = EncBlock(inc=self.base_width*2, ouc=self.base_width*4)
        self.enc4 = EncBlock(inc=self.base_width*4, ouc=base_width*8)
        self.enc5 = StemBlock(inc=self.base_width*8, ouc=self.base_width*8) 

        #Decoder Part
        self.up1 = DecUpBlock(inc=self.base_width*8, ouc=self.base_width*8) 
        self.dec1 = DecDownBlock(inc=self.base_width*8, ouc=self.base_width*4)

        self.up2 = DecUpBlock(inc=self.base_width*4, ouc=self.base_width*4)
        self.dec2 = DecDownBlock(inc=self.base_with*4, ouc=self.base_width*2)

        self.up3 = DecUpBlock(inc=self.base_width*2, ouc=self.base_with*2)
        self.dec3 = DecDownBlock(inc=self.base_with*2, ouc=self.base_width)

        self.up4 = DecUpBlock(inc=self.base_width, ouc=self.base_width)
        self.dec4 = DecDownBlock(inc=self.base_width, ouc=self.base_width)

        self.fin_out = nn.Conv2d(base_width, self.fin_ouc, kernel_size=3, padding=1)

    def forward(self, x):
        #Encode
        d1 = self.enc1(x)
        d2 = self.enc2(x)
        d3 = self.enc3(x)
        d4 = self.enc4(x)
        d5 = self.enc5(x)

        #Decode
        u1 = self.up1(d5)
        de1 = self.dec1(u1)

        u2 = self.up2(de1)
        de2 = self.dec1(u2)

        u3 = self.up3(de2)
        de3 = self.dec3(u3)

        u4 = self.up4(de3)
        de4 = self.dec(u4)

        output = self.fin_out(de4)
        return output

        
        
        

        