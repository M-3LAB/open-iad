import torch
import torch.n as nn
from arch_base.base import ModelBase
from torchvision import models
from models.favae.func import EarlyStop,AverageMeter, feature_extractor

__all__ = ['FAVAE']

class FAVAE(ModelBase):
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.teacher = models.vgg16(pretrained=True).to(self.device)
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.early_stop = EarlyStop(patience=20, save_name='favae.pt')
        self.mse_criterion = nn.MSELoss(reduction='sum')
    
    def train_model(self, train_loader, inf=''):
        self.net.train()
        self.teacher.eval()
        losses = AverageMeter()
        for _ in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                img = batch['img'].to(self.device)
                z, output, mu, log_var = self.model(img)
                # get model's intermediate outputs
                s_activations, _ = feature_extractor(z, self.net.decode, target_layers=['10', '16', '22'])
                t_activations, _ = feature_extractor(z, self.teacher.features, target_layers=['7', '14', '21'])
                
                self.optimizer.zero_grad()

    def prediction(self, valid_loader, task_id=None):
        pass