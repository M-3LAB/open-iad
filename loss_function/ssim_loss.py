import torch
import torch.nn as nn
from math import exp

__all__ = ['SSIMLoss']

class SSIMLoss(nn.Module):
    def __init__(self, window_size, size_average, sigma):
        super(SSIMLoss, self).__init__()

        self.window_size = window_size
        self.size_average = size_average
        self.sigma = sigma
        

    def create_window(self, channel=1):
        window_1d = self.gaussian(self.window_size, 1.5).unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
        window = window_2d.expand(channel, 1, self.window_size, self.window_size).contiguous()
        return window

    def gaussian(self):
        gauss = torch.Tensor([exp(-(x - self.window_size//2)**2/float(2* self.sigma**2)) for x in range(self.window_size)])
        return gauss/gauss.sum()

    def calculate_ssim(self):
        pass

    def forward(self, x):
        pass