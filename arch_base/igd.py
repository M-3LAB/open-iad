import torch
import torch.nn as nn
from models.igd.ssim_module import *
from models.igd.mvtec_module import *


__all__ = ['IGD']

class IGD():
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        self.config = config
        self.device = device
        self.file_path = file_path
        self.scheduler = scheduler
        self.generator = self.net['g'].to(self.device)
        self.discriminator = self.net['d'].to(self.device)

    def init_c(self, data_loader, generator, eps=0.1):
        generator.c = None
        c = torch.zeros(1, self.config['latent_dimension']).to(self.device)
        generator.eval()
        n_samples = 0
        with torch.no_grad():
            for index, (images, label) in enumerate(data_loader):
                # get the inputs of the batch
                img = images.to(self.device)
                outputs = generator.encoder(img)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        
        c /= n_samples 

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


    def init_sigma(self):
        pass
    
    def train_model(self, train_loaders, inf=''):
        self.generator.train()
        self.discriminator.train()

        for param in self.generator.pretrain.parameters():
            param.requires_grad = False
        
        
            

    def prediction(self, valid_loader):
        self.generator.eval()
        self.discriminator.eval()