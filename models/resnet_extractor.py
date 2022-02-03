import torch
import torch.nn as nn
from torchvision import models

__all__ = ['ResNetExtractor']

class ResNetExtractor(nn.Module):
    def __init__(self, device, backbone_name='resnet18'):
        super(ResNetExtractor).__init__()
        assert backbone_name in ['resnet18', 'wide_resnet50']  

        self.device = device

        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=True, progress=True).to(self.device)
        elif backbone_name == 'wide_resnet50':
            self.backbone = models.wide_resnet50_2(pretrained=True, progress=True).to(self.device)
    
    def forward(self, input):
        pass
