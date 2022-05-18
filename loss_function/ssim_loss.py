import torch
import torch.nn as nn
from math import exp
import torch.nn.functional as F

__all__ = ['SSIMLoss']

class SSIMLoss(nn.Module):
    def __init__(self, window_size, size_average, sigma, 
                 channel_number=1, device=None):
        super(SSIMLoss, self).__init__()

        self.window_size = window_size
        self.size_average = size_average
        self.sigma = sigma
        self.channel_number = channel_number
        self.window = self.create_window().to(device)
        
    def create_window(self):
        window_1d = self.gaussian(self.window_size, 1.5).unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
        window = window_2d.expand(self.channel_number, 1, self.window_size, self.window_size).contiguous()
        return window

    def gaussian(self):
        gauss = torch.Tensor([exp(-(x - self.window_size//2)**2/float(2* self.sigma**2)) for x in range(self.window_size)])
        return gauss/gauss.sum()

    def calculate_ssim(self, img_a, img_b, window):
        padding = self.window_size // 2

        mu1 = F.conv2d(img_a, window, padding=padding, groups=self.channel)
        mu2 = F.conv2d(img_b, window, padding=padding, groups=self.channel)


    def forward(self, img_a, img_b):
        _, channel, _, _ = img_a.size()