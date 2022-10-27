import torch
import torch.n as nn
from arch_base.base import ModelBase
from torchvision import models
from models.favae.func import EarlyStop,AverageMeter

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
        
        self.early_stop = EarlyStop(patience=20, save_name='favae.pt')
    
    def train_model(self, train_loaders, inf=''):
        self.net.train()
        self.teacher.eval()
        losses = AverageMeter()

    def prediction(self, valid_loader, task_id=None):
        pass