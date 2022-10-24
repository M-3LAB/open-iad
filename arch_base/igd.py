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

    def init_sigma(self, data_loader, generator, sig_f=1):
        generator.sigma = None
        generator.eval()
        tmp_sigma = torch.tensor(0.0, dtype=torch.float).to(self.device)
        n_samples = 0
        with torch.no_grad():
            for index, (images, label) in enumerate(data_loader):
                img = images.to(self.device)
                latent_z = self.generator.encoder(img)
                diff = (latent_z - self.generator.c) ** 2
                tmp = torch.sum(diff.detach(), dim=1)
                if (tmp.mean().detach() / sig_f) < 1:
                    tmp_sigma += 1
                else:
                    tmp_sigma += tmp.mean().detach() / sig_f
                n_samples += 1
        tmp_sigma /= n_samples
        return tmp_sigma
    
    def train_model(self, train_loaders, inf=''):
        for task_idx, train_loader in enumerate(train_loaders):

            print('run task: {}'.format(self.config['train_task_id'][task_idx]))
            AUC_LIST = [] 
            global test_auc
            test_auc = 0
            self.generator.c = None
            self.generator.sigma = None

            BEST_AUC = 0

            self.generator.train()
            self.discriminator.train()

            for param in self.generator.pretrain.parameters():
                param.requires_grad = False
        
            START_ITER = 0
            for _ in range(self.config['num_epochs']):
                for batch_id, batch in enumerate(train_loader):
                    train_size = len(train_loader)
                    END_ITER = int(train_size / self.config['batch_size'] * self.config['max_epoch']) 

                    self.generator.c = self.init_c(train_loader, self.generator)
                    self.generaotr.c.requries_grad = False
                    self.generator.sigma = self.init_sigma(train_loader, self.generator)


        
        
        
        
            

    def prediction(self, valid_loader):
        self.generator.eval()
        self.discriminator.eval()