import torch
import torch.nn as nn
from sklearn.random_projection import SaprseRandomProjection 
from sklearn.neighbors import NearestNeighbors
from sampling.kcenter_greedy import KCenterGreedy

__all__ = ['PatchCoreTrainer']

class PatchCoreTrainer(object):

    def __init__(self, config, train_loader, test_loader):

        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.random_projector = SaprseRandomProjection(n_components='auto',
                                                       eps=0.9)
        
        self.selector = KCenterGreedy()
    
    def train(self):
        pass

    def predict(self):
        pass
