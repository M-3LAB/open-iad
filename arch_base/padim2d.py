import torch
import torch.nn as nn
from models.padim.padim import PaDim 

__all__ = ['PatchCore2D']

class PaDim2D():
    def __init__(self, config, train_loader, valid_loader, device):
        
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device

        self.model = SPADE().to(self.device)


    def train_epoch(self, inf=''):
        for task_idx, train_loader in enumerate(self.train_loader):
            print('run task: {}'.format(task_idx))

            for i, batch in enumerate(train_loader):
                if self.config['debug'] and i > self.batch_limit:
                    break

                x, y, mask, task_id = batch
                
                pass


    def prediction(self):
        acc = 0

        for i, batch in enumerate(self.valid_loader):
            x, y, mask, task_id = batch
                
            pass

        return acc