import torch
import torch.nn as nn

__all__ = ['FeatureExtractor']

class FeatureExtractor(nn.Module):
    def __init__(self, backbone, layers):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        # collect the output dimension from each layer
        self.output_dims = []
        self._features = {layer: torch.empty(0) for layer in self.layers}

    def get_layer_features(self, layer_id):
        pass

    def forward(self, input):
        pass
        