import torch
import torch.nn as nn

__all__ = ['NetGraphCore']

class NetGraphCore(nn.module):
    def __init__(self):
        super().__init__()
