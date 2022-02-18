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

        self.random_projector = SaprseRandomProjection(n_components='auto',
                                                       eps=0.9)
        #Model 
        self.model = PatchCore(backbone_name=self.config.backbone_name,
                               device=self.device,
                               layer_hook=self.config.layer_hook,
                               layer_indices=self.config.layer_indices,
                               channel_indices=self.channel_indices) 
        # loss function

        # optimizer

        # lr scheduler

        
    def train(self):
        pass

    def predict(self):
        pass
