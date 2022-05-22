import torch
import torch.nn as nn
from math import exp
import torch.nn.functional as F

__all__ = ['SSIMLoss']

class SSIMLoss(nn.Module):
    def __init__(self, window_size, size_average, sigma, 
                 channel_number=1, device=None, value_range=None):
        super(SSIMLoss, self).__init__()

        self.window_size = window_size
        self.size_average = size_average
        self.sigma = sigma
        self.channel_number = channel_number
        self.window = self.create_window().to(device)
        self.value_range = self.define_val_range(predefined_value_range=value_range)
        
    def create_window(self):
        window_1d = self.gaussian(self.window_size, 1.5).unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
        window = window_2d.expand(self.channel_number, 1, self.window_size, self.window_size).contiguous()
        return window

    def gaussian(self):
        gauss = torch.Tensor([exp(-(x - self.window_size//2)**2/float(2* self.sigma**2)) for x in range(self.window_size)])
        return gauss/gauss.sum()

    def define_val_range(self, predefined_value_range=None):
        if predefined_value_range is None: 
            pass
        else:
            return predefined_value_range

    def calculate_ssim(self, img_a, img_b, window):
        padding = self.window_size // 2

        mu1 = F.conv2d(img_a, window, padding=padding, groups=self.channel)
        mu2 = F.conv2d(img_b, window, padding=padding, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img_a * img_b, window, padding=padding, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img_a * img_b, window, padding=padding, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img_a * img_b, window, padding=padding, groups=self.channel) - mu1_mu2

        c1 = (0.01 * self.value_range) ** 2
        c2 = (0.03 * self.value_range) ** 2

        v1 = 2.0 * sigma12 + c2
        v2 = sigma1_sq + sigma2_sq + c2
        cs = torch.mean(v1 / v2)  # contrast sensitivity


    def forward(self, img_a, img_b):
        _, channel, _, _ = img_a.size()