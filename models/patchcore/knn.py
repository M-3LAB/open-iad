import torch
import torch.nn as nn

__all__ = ['KNNExtractor']

class KNNExtractor(nn.Module):
    def __init__(self):
        super().__init__()