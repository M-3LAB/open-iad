import torch
import torch.nn as nn
from arch_base.base import ModelBase


__all__ = ['GraphCore']

class GraphCore(ModelBase):

    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        super().__init__(config, device)

        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def forward(self, x):
        pass