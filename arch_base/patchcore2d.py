import torch
import torch.nn as nn
from sklearn.random_projection import SaprseRandomProjection 
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

        self.random_projector = SaprseRandomProjection(n_components='auto',
                                                       eps=0.9)
        #Model 
        self.model = PatchCore(backbone_name=self.config.backbone_name,
                               device=self.device,
                               layer_hook=self.config.layer_hook,
                               layer_indices=self.config.layer_indices,
                               channel_indices=self.config.channel_indices) 
        
        # loss function
        self.criterion = torch.nn.MSELoss(reduction='sum').to(self.device)

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['lr'],
                                         momentum=self.config['momentum'], 
                                         weight_decay=self.config['weight_decay']) 

        # lr scheduler

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