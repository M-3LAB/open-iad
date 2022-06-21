import torch
import torch.nn as nn
import torch.nn.functional as F
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
        anomaly_map = patch_scores[:, 0].reshape((batch_size, 1, width, height))
        anomaly_map = F.interpolate(anomaly_map, size=(self.input_size[0], self.input_size[1]))
        
        kernel_size = 2 * int(4.0 * self.sigma + 0.5) + 1
        anomaly_map = gaussian_blur2d(anomaly_map, (kernel_size, kernel_size), sigma=(self.sigma, self.sigma))

        return anomaly_map

    @staticmethod 
    def compute_anomaly_score(patch_scores):
        """Compute image-level anomaly score

        Args:
            patch_scores (torch.Tensor): 
        
        Returns: 
            image-level scores (torch.Tensor)
        """
        #TODO: Why?
        max_scores = torch.argmax(patch_scores[:, 0])
        confidence = torch.index_select(patch_scores, 0, max_scores)
        weights = 1 - (torch.max(torch.exp(confidence)) / torch.sum(torch.exp(confidence)))
        score = weights * torch.max(patch_scores[:, 0])
        return score

    def __call__(self, **kwargs):
        """Returns anomaly_map and anomaly_score.

        Expects `patch_scores` keyword to be passed explicitly
        Expects `feature_map_shape` keyword to be passed explicitly

        Example
        >>> anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)
        >>> map, score = anomaly_map_generator(patch_scores=numpy_array, feature_map_shape=feature_map_shape)

        Raises:
            ValueError: If `patch_scores` key is not found

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: anomaly_map, anomaly_score
        """

        if "patch_scores" not in kwargs:
            raise ValueError(f"Expected key `patch_scores`. Found {kwargs.keys()}")

        if "feature_map_shape" not in kwargs:
            raise ValueError(f"Expected key `feature_map_shape`. Found {kwargs.keys()}")

        patch_scores = kwargs["patch_scores"]
        feature_map_shape = kwargs["feature_map_shape"]

        anomaly_map = self.compute_anomaly_map(patch_scores, feature_map_shape)
        anomaly_score = self.compute_anomaly_score(patch_scores)
        return anomaly_map, anomaly_score