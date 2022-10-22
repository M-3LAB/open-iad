from __future__ import nested_scopes
import torch
from torch import nn

__all__ = ['PatchCore2D']

class _CSFlow(nn.Module):
    def __init__(self, args, net, optimizer, scheduler):
        super(_CSFlow, self).__init__()
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net

    def forward(self, epoch, inputs, labels, one_epoch_embeds, task_wise_mean, task_wise_cov, t):
        self.optimizer.zero_grad()
        embeds, z, log_jac_det = self.net(inputs)
        # yy, rev_y, zz = self.net.revward(inputs)
        loss = torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - log_jac_det) / z.shape[1]

        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step(epoch)



class CSFlow():
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net

        self.backbone = _CSFlow(config, self.net, optimizer, scheduler)

        self.features = [] 

        source_domain = ''
        if self.config['continual']:
            for i in self.config['train_task_id']:  
                source_domain = source_domain + str(self.config['train_task_id'][i])
        else:
            source_domain = str(self.config['train_task_id'][0])

    

        
    def train_epoch(self, train_loaders, inf=''):
        epoch = 1
        one_epoch_embeds = []
        task_wise_mean, task_wise_cov = [], []
        self.backbone.train()
        # When num_task is 15, per task means per class
        for task_idx, train_loader in enumerate(train_loaders):
            print('run task: {}'.format(self.config['train_task_id'][task_idx]))
            for _ in range(self.config['num_epochs']):
                for batch_id, batch in enumerate(train_loader):
                    inputs = batch['img'].to(self.device)
                    labels = batch['label'].to(self.device)

                    # Extract features from backbone
                    self.features.clear()
                    self.backbone(epoch, inputs, labels, one_epoch_embeds, task_wise_mean, task_wise_cov, task_idx)


    def prediction(self, valid_loader):

        self.backbone.eval()



        return pixel_auroc, img_auroc



        
      

