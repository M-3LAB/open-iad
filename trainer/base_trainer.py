from turtle import forward
import torch
import torch.nn as nn


__all__ = ['BaseTrainer']

#TODO: Jinbao Not Finished Yet
class BaseTrainer(nn.Module):

    def __init__(self):
        super(BaseTrainer).__init__()
    
    def forward(self, x):
        pass
