import torch
import torch.nn as nn
from kornia.filters import gaussian_blur2d

__all__ = ['AnomalyMapGenerator']

class AnomalyMapGenerator:
    def __init__(self, input_size, sigma):
        self.input_size = input_size
        self.sigma = sigma

    def compute_anomaly_map(self, patch_scores, feature_map_shape):
        """Pixel level heat map

        Args:
            patch_scores (torch.Tensor): Patch level anomaly scores 
            feature_map_shape (torch.size): 2D feature map shape 
        
        Returns: 
            pixel level anomaly maps (torch.tensor)
        """
        height, width = feature_map_shape
        batch_size = len(patch_scores) // (width * height)
        return anomaly_map

    def compute_anomaly_score(patch_scores):
        pass

    def __call__(self):
        pass