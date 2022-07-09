import torch
import torch.nn as nn
from models.reverse.resnet import * 

__all__ = ['Reverse']

class Reverse():
    def __init__(self, config, train_loaders, valid_loaders,
                 device, file_path, train_fewshot_loaders=None):

        self.config = config
        self.train_loaders = train_loaders
        self.valid_loaders = valid_loaders
        self.device = device
        self.file_path = file_path
        self.train_fewshot_loaders = train_fewshot_loaders

    def train_epoch(self):
        pass

    def prediction(self):
        pass