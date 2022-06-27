import torch
import torch.nn as nn
from torchvision import models
#from models.patchcore.patchcore import PatchCore

__all__ = ['PatchCore2D']

class PatchCore2D():
    def __init__(self, config, train_loaders, valid_loaders, device):
        
        self.config = config
        self.train_loaders = train_loaders
        self.valid_loaders = valid_loaders
        self.device = device

        # Backbone model
        if config['backbone'] == 'resnet18':
            self.backbone = models.resnet18(pretrained=True, progress=True).to(self.device)
        elif config['backbone'] == 'wide_resnet50':
            self.backbone = models.wide_resnet50_2(pretrained=True, progress=True).to(self.device)
        else:
            raise NotImplementedError('This Pretrained Model Not Implemented Error')
        
        self.backbone.eval()
       
        
    def train_epoch(self, inf=''):

        for epoch in self.config['num_epoch']:
            for task_idx, train_loader in enumerate(self.train_loaders):
                print('run task: {}'.format(task_idx))
                for batch_id, batch in enumerate(train_loader):
                    if self.config['debug'] and batch_id > self.batch_limit:
                        break
                    img = batch['image'].to(self.device)

                              
                    
                    
    def prediction(self):
        pass
      

