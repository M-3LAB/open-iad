import torch.nn.modules.conv as conv
import torch.nn as nn
import torch

__all__ = ['CoordConv2d', 'AddCoords']

class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, device, stride=1,
                padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, device, with_r)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                            kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        out = self.addcoords(input_tensor)
        #self.conv.to(self.device)
        out = self.conv(out)

        return out

class AddCoords(nn.Module):
    def __init__(self, rank, device, with_r=False): 
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.device = device

    def forward(self, input_tensor):
        batch_size_shape, _, dim_y, dim_x = input_tensor.shape
        xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

        xx_range = torch.arange(dim_y, dtype=torch.int32)
        yy_range = torch.arange(dim_x, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (dim_y - 1)
        yy_channel = yy_channel.float() / (dim_x - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

        if torch.cuda.is_available:
            input_tensor = input_tensor.to(self.device)
            xx_channel = xx_channel.to(self.device)
            yy_channel = yy_channel.to(self.device)
        out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out = torch.cat([out, rr], dim=1)

        return out