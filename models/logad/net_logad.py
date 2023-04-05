import torch
import torch.nn as nn

__all__ = ['NetLogAD']

class NetLogAD(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        pass