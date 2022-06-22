import torch
import torch.nn as nn
from sklearn.random_projection import SparseRandomProjection 
from sklearn.neighbors import NearestNeighbors
from sampling.kcenter_greedy import KCenterGreedy
from models.patchcore.patchcore import PatchCore

__all__ = ['PatchCore2D']

class PatchCore2D():
    def __init__(self, config, train_loader, valid_loader, device):
        
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device

        #Model 
        self.model = PatchCore(input_size=self.config['input_size'],
                               backbone=self.config['backbone'],
                               layers=self.config['layers'],
                               num_neighbours=self.config['num_neighbours']).to(self.device)
        
        self.coreset_sampling_ratio = self.config['sampling_ratio'] 
        self.embeddings = []
        
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