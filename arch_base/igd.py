import torch
import torch.nn as nn
from models.igd.ssim_module import *
from models.igd.mvtec_module import *

__all__ = ['IGD']

class IGD():
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        self.config = config
        self.device = device
        self.file_path = file_path
        self.scheduler = scheduler
        self.generator = self.net['g'].to(self.device)
        self.discriminator = self.net['d'].to(self.device)

    
    def train_model(self, train_loaders, inf=''):
        for param in self.generator.pretrain.parameters():
            param.requires_grad = False
            

    def prediction(self, valid_loader):
        pass