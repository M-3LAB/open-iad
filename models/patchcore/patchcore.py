import torch
import torch.nn as nn
import numpy as np
from models.common.feat import FeatureExtractor

__all__ = ['PatchCore']

class PatchCore(nn.Module):

    def __init__(self, input_size, backbone_name, device, layers, num_neighbours):

        super(PatchCore).__init__()

        """
        Args:
            backbone: The name of the desired backbone, i.e., ['resnet18', 'wide_resnet'] 
            device: GPU
        """

        self.device = device
        self.backbone_name = backbone_name
        self.layers = layers
        self.input_size = input_size
        
    def forward(self, x):
        pass

        
