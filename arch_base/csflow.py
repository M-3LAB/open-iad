from __future__ import nested_scopes
import torch
from torch import nn
import numpy as np

from arch_base.base import ModelBase
from metrics.common.np_auc_precision_recall import np_get_auroc


__all__ = ['CSFlow']

class _CSFlow(nn.Module):
    def __init__(self, args, net, optimizer, scheduler):
        super(_CSFlow, self).__init__()
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net
        self.net.feature_extractor.eval()
        
    def forward(self, epoch, inputs):
        self.optimizer.zero_grad()
        embeds, z, log_jac_det = self.net(inputs)
        # yy, rev_y, zz = self.net.revward(inputs)
        loss = torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - log_jac_det) / z.shape[1]

        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step(epoch)


class CSFlow(ModelBase):
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
        self.model = _CSFlow(self.config, self.net, optimizer, scheduler).to(self.device)
        
    def train_model(self, train_loaders, inf=''):
        self.net.density_estimator.train()

        for task_idx, train_loader in enumerate(train_loaders):
            print('run task: {}'.format(self.config['train_task_id'][task_idx]))

            for epoch in range(self.config['num_epochs']):
                for batch_id, batch in enumerate(train_loader):
                    inputs = batch['img'].to(self.device)
                    self.model(epoch, inputs)


    def prediction(self, valid_loader, task_id=None):
        self.net.eval()
        pixel_auroc, img_auroc = 0, 0

        test_z, test_labels = [], []
        with torch.no_grad():
            for batch_id, batch in enumerate(valid_loader):
                inputs = batch['img'].to(self.device)
                labels = batch['label'].to(self.device)

                _, z, jac = self.net(inputs)
                z = z[..., None].cpu().data.numpy()
                score = np.mean(z ** 2, axis=(1, 2))
                test_z.append(score)
                test_labels.append(labels.cpu().data.numpy())

            test_labels = np.concatenate(test_labels)
            is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])
            anomaly_score = np.concatenate(test_z, axis=0)
            img_auroc = np_get_auroc(is_anomaly, anomaly_score)

        return pixel_auroc, img_auroc