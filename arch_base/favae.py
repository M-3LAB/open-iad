import torch
import torch.n as nn
from arch_base.base import ModelBase
from torchvision import models
from models.favae.func import EarlyStop,AverageMeter, feature_extractor, print_log

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
        for epoch in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                img = batch['img'].to(self.device)
                z, output, mu, log_var = self.model(img)
                # get model's intermediate outputs
                s_activations, _ = feature_extractor(z, self.net.decode, target_layers=['10', '16', '22'])
                t_activations, _ = feature_extractor(z, self.teacher.features, target_layers=['7', '14', '21'])

                self.optimizer.zero_grad()
                mse_loss = self.mse_criterion(output, img)
                kld_loss = 0.5 * torch.sum(-1 - log_var + torch.exp(log_var) + mu**2)
                for i in range(len(s_activations)):
                    s_act = self.net.adapter[i](s_activations[-(i + 1)])
                    mse_loss += self.mse_criterion(s_act, t_activations[i])
                loss = mse_loss + self.config['kld_weight'] * kld_loss
                losses.update(loss.sum().item(), img.size(0))

                loss.backward()
                self.optimizer.step()
            
            infor = '\r{}[Epoch {}/{}] Loss: {:.6f}'.format(inf, epoch, self.config['num_epochs'],
                                                            losses.avg) 
            print(infor, flush=True, end='')

    def prediction(self, valid_loader, task_id=None):
        self.net().eval()
        self.teacher.eval()
        pass