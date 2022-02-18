import torch
import torch.nn as nn

__all__ = ['PatchCoreTrainer']

class PatchCoreTrainer(object):

    def __init__(self, config, train_loader, test_loader):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
    
    def train(self):
        pass

    def predict(self):
        pass
