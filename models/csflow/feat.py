import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

__all__ = ['FeatureExtractor']

class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super(FeatureExtractor, self).__init__()

        if self.backbone == 'efficient_net':
            self.feature_extractor = EfficientNet.from_pretrained()