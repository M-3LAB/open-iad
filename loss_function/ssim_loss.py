import torch
import torch.nn as nn

__all__ = ['SSIMLoss']

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        pass

    def create_window(self, windown_size, channel=1):
        pass

    def gaussian(self):
        pass

    def calculate_ssim(self):
        pass

    def forward(self, x):
        pass