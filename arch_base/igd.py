import torch
import torch.nn as nn

class IGD():
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
    
    
    def train_model(self, train_loaders, inf=''):
        pass