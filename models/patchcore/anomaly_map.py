import torch
import torch.nn as nn

__all__ = ['AnomalyMapGenerator']

class AnomalyMapGenerator:
    def __init__(self, input_size, sigma):
        self.input_size = input_size
        self.sigma = sigma

    def compute_anomaly_map(self):
        pass

    def compute_anomaly_score(self):
        pass

    def __call__(self):
        pass