import torch
import torch.n as nn
from arch_base.base import ModelBase

__all__ = ['FAVAE']

class FAVAE(ModelBase):
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        self.config = config
    
    def train_model(self, train_loaders, inf=''):
        pass