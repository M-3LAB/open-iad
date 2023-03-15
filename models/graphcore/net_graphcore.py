import torch
import torch.nn as nn
from models.graphcore.pyramid_vig import *
from models.graphcore.vig import *

__all__ = ['NetGraphCore']

class NetGraphCore(nn.module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    
    #def forward(self, x):
    #    pass
