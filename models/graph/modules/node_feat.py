import torch
import torch.nn as nn

__all__ = ['NodeFeat']

class NodeFeat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, node_inp):
        pass