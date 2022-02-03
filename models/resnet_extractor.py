import torch
import torch.nn as nn

__all__ = ['ResNetExtractor']

class ResNetExtractor(nn.Module):
    def __init__(self):
        super().__init__()
