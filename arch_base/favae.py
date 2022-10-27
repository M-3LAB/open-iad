import torch
import torch.n as nn
from arch_base.base import ModelBase
from torchvision import models
from models.favae.func import EarlyStop

__all__ = ['FAVAE']

class FAVAE(ModelBase):
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.teacher = models.vgg16(pretrained=True).to(self.device)
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def train_model(self, train_loaders, inf=''):
        pass

    def prediction(self, valid_loader, task_id=None):
        pass