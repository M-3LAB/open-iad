import torch
import torch.nn as nn
from sklearn.random_projection import SaprseRandomProjection 
from sklearn.neighbors import NearestNeighbors
from sampling.kcenter_greedy import KCenterGreedy
from models.patchcore.patchcore import PatchCore

__all__ = ['PatchCoreTrainer']

class PatchCoreTrainer(object):

    def __init__(self, config, device, train_loader, test_loader):

        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        self.random_projector = SaprseRandomProjection(n_components='auto',
                                                       eps=0.9)
        #Model 
        self.model = PatchCore(backbone_name=self.config.backbone_name,
                               device=self.device,
                               layer_hook=self.config.layer_hook,
                               layer_indices=self.config.layer_indices,
                               channel_indices=self.channel_indices) 
        # loss function
        self.criterion = torch.nn.MSELoss(reduction='sum').to(self.device)

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['lr'],
                                         momentum=self.config['momentum'], 
                                         weight_decay=self.config['weight_decay']) 

        # lr scheduler

        
    def train(self):
        pass

    def predict(self):
        pass
