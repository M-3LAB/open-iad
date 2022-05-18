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
        pass

    def gaussian(self):
        gauss = torch.Tensor([exp(-(x - self.window_size//2)**2/float(2* self.sigma**2)) for x in range(self.window_size)])
        return gauss/gauss.sum()

    def calculate_ssim(self):
        pass

    def forward(self, x):
        pass