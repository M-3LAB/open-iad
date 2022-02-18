import torch
import torch.nn as nn

__all__ = ['PaDim']

class PaDim(nn.Module):

    def __init__(self, device, backbone_name):
        super(PaDim).__init__()

        self.device = device
        self.backbone_name = backbone_name
    
    def forward(self, x):
        pass
