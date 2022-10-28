import torch
import torch.nn as nn
from arch_base.base import ModelBase
from models.fastflow.func import AverageMeter

__all__ = ['FastFlow']

class FastFlow(ModelBase):
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_model(self, train_loader, inf=''):
        self.net.train()
        loss_meter = AverageMeter() 
        for _ in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                img = batch['img'].to(self.device)
    
    def prediction(self, valid_loader, task_id=None):
        pass