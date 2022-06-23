import torch
import torch.nn as nn
from models.patchcore.patchcore import PatchCore
import logging

__all__ = ['PatchCore2D']

class PatchCore2D():
    def __init__(self, config, train_loaders, valid_loaders, device):
        
        self.config = config
        self.train_loaders = train_loaders
        self.valid_loaders = valid_loaders
        self.device = device

        #Model 
        self.model = PatchCore(input_size=self.config['input_size'],
                               backbone=self.config['backbone'],
                               layers=self.config['layers'],
                               num_neighbours=self.config['num_neighbours']).to(self.device)
        
        self.coreset_sampling_ratio = self.config['sampling_ratio'] 
        self.embeddings = []
        
    def train_epoch(self, inf=''):

        # Extract features for each image 
        self.model.feature_extractor.eval()

        for epoch in self.config['num_epoch']:
            for task_idx, train_loader in enumerate(self.train_loaders):
                print('run task: {}'.format(task_idx))
                for batch_id, batch in enumerate(train_loader):
                    if self.config['debug'] and batch_id > self.batch_limit:
                        break
                    img = batch['image'].to(self.device)

                    embedding = self.model(img)
                    self.embeddings.append(embedding)                    
                    
                    
    def prediction(self):
        self.model.eval()
        pass
