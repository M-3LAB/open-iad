from __future__ import nested_scopes
import torch
from torch import nn
import numpy as np
import argparse

from arch.base import ModelBase
from models.net_csflow.net_csflow import NetCSFlow
from optimizer.optimizer import get_optimizer

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
    def __init__(self, config):
        super(CSFlow, self).__init__(config)
        self.config = config
        args = argparse.Namespace(**self.config)
        self.net = NetCSFlow(args)
        self.optimizer = get_optimizer(self.config, self.net.density_estimator.parameters())
        self.model = _CSFlow(args, self.net, self.optimizer, self.scheduler).to(self.device)
        
    def train_model(self, train_loader, task_id, inf=''):
        self.net.density_estimator.train()

        for epoch in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                inputs = batch['img'].to(self.device)
                self.model(epoch, inputs)

    def prediction(self, valid_loader, task_id):
        self.net.eval()
        self.clear_all_list()

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
                self.img_path_list.append(batch['img_src'])

            test_labels = np.concatenate(test_labels)
            is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])
            anomaly_score = np.concatenate(test_z, axis=0)
            self.img_gt_list = is_anomaly
            self.img_pred_list = anomaly_score