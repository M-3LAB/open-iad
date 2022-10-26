from arch_base.base import ModelBase
import torch
import torch.nn as nn
from arch_base.base import ModelBase

__all__ = ['DevNet']

class DevNet(ModelBase):
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def train_model(self, train_loaders, inf=''):
        pass

    def prediction(self, valid_loder, task_id=None):
        pass