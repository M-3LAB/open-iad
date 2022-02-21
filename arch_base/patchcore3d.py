import torch
import numpy as np




__all__ = ['PatchCore3D']



class PatchCore3D():
    def __init__(self, config, train_loader, valid_loader,
                 device, file_path, batch_limit_weight=1.0):
        
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.file_path = file_path
        self.batch_limit_weight = batch_limit_weight
        self.batch_limit = 2

    def train_epoch(self, inf=''):
        for i, batch in enumerate(self.train_loader):
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