from msilib.schema import Feature
import torch
import torch.nn as nn
import numpy as np
from models.common.feat import FeatureExtractor
from models.patchcore.anomaly_map import AnomalyMapGenerator

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
        self.num_neighbours = num_neighbours

        self.feature_extractor = FeatureExtractor(backbone=self.bachbone(pretrained=True), layers=self.layers) 
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)

        self.register_buffer("memory_bank", torch.Tensor())
        self.memory_bank: torch.Tensor
    
    def generate_embedding(self):
        pass

    def reshape_embedding(self):
        pass
        
        
    def forward(self, x):

        """
        Return embedding during training
        """
        with torch.no_grad():
            features = self.feature_extractor(x)

        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}

        
