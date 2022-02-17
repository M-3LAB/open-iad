import torch
import torch.nn as nn
import numpy as np
from models.resnet_extractor import ResNetExtractor 

__all__ = ['PatchCore']

class PatchCore(torch.nn.Module):

    def __init__(self, backbone_name, device, layer_hook, layer_indices,
                 channel_indices, corest):

        super(PatchCore).__init__()

        """
        Args:
            backbone: The name of the desired backbone, i.e., ['resnet18', 'wide_resnet'] 
            device: GPU
            channel indices: A tensor with the de
        """

        self.device = device
        self.backbone_name = backbone_name
        self.feat_extractor = ResNetExtractor(device=self.device, 
                                              backbone_name=self.backbone_name).to(self.device) 

        self.layer_hook = torch.nn.AvgPool2d(3, 1, 1) if layer_hook is None else layer_hook
        self.layer_indices = [1, 2] if layer_indices is None else layer_indices 

        self.corest = corest
        self.channel_indices = channel_indices
        
     

    def forward(self, x):
        pass