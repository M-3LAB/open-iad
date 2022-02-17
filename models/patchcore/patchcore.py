import torch
import torch.nn as nn
import numpy as np
from models.resnet_extractor import ResNetExtractor 

__all__ = ['PatchCore']

class PatchCore(torch.nn.Module):
    def __init__(self, backbone_name, device):
        super(PatchCore).__init__()

        self.device = device
        self.backbone_name = backbone_name
        self.feat_extractor = ResNetExtractor(device=self.device, 
                                              backbone_name=self.backbone_name) 
        
     
    
    def forward(self, x):
        pass