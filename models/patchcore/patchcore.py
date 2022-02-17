import torch
import torch.nn as nn
import numpy as np
from models.resnet_extractor import ResNetExtractor 

__all__ = ['PatchCore']

class PatchCore(torch.nn.Module):
    def __init__(self, backbone_name, device, layer_hook, layer_indices):
        super(PatchCore).__init__()

        self.device = device
        self.backbone_name = backbone_name
        self.feat_extractor = ResNetExtractor(device=self.device, 
                                              backbone_name=self.backbone_name).to(self.device) 
        self.layer_hook = layer_hook
        self.layer_indices = layer_indices

    def forward(self, x):
        pass