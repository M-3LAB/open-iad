import torch
import numpy as np

__all__ = ['PointCore']

class PointCore():
    def __init__(self, config, train_loader, valid_loader, device):
        
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device

    def train_epoch(self, inf=''):
        for task_idx, train_loader in enumerate(self.train_loader):
            print('run task: {}'.format(task_idx))

            for i, batch in enumerate(train_loader):
                if self.config['debug'] and i > self.batch_limit:
                    break

                x, y, mask, task_id, xyz = batch
                
                pass


    def evaluation(self):
        acc = 0

        for i, batch in enumerate(self.valid_loader):
            x, y, mask, task_id, xyz = batch
                
            pass

        return acc