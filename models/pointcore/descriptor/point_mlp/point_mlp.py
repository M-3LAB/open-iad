import torch
import torch.nn as nn

__all__ = ['PointMLP']

class PointMLP(nn.Module):
    def __init__(self, num_affinity_points):
        super(PointMLP).__init__()
        self.num_affinity_points = num_affinity_points

    def forward(self, x):
        pass