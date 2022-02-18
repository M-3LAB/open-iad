import torch
import torch.nn as nn
import numpy as np
from models.resnet_extractor import ResNetExtractor 

__all__ = ['PatchCore']

class PatchCore(nn.Module):

    def __init__(self, backbone_name, device, layer_hook, layer_indices,
                 channel_indices):

        super(PatchCore).__init__()

        """
        Args:
            backbone: The name of the desired backbone, i.e., ['resnet18', 'wide_resnet'] 
            device: GPU
            channel indices: A tensor with the desired channel indices to extract from the backbone
            layer_hook: A function layer_hook: A function that runs on each layer of the resnet before
                concatenating them. 
            layer_indices: A list of indices with the desired layers to include in the
                embedding vectors.
        """

        self.device = device
        self.backbone_name = backbone_name
        self.feat_extractor = ResNetExtractor(device=self.device, 
                                              backbone_name=self.backbone_name).to(self.device) 

        self.layer_hook = torch.nn.AvgPool2d(3, 1, 1) if layer_hook is None else layer_hook
        self.layer_indices = [1, 2] if layer_indices is None else layer_indices 

        self.channel_indices = channel_indices.to(self.device)
        
    def forward(self, x):

        embedding_feat = self.feat_extractor(x, channel_indices=self.channel_indices, 
                                             layer_hook=self.layer_hook, 
                                             layer_indices=self.layer_indices)

        return embedding_feat