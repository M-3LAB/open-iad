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
        super(CSFlow, self).__init__(config, device, file_path, net, optimizer, scheduler)
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
        self.model = _CSFlow(self.config, self.net, optimizer, scheduler).to(self.device)
        
    def train_model(self, train_loader, task_id, inf=''):
        self.net.density_estimator.train()

        for epoch in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                inputs = batch['img'].to(self.device)
                self.model(epoch, inputs)


    def prediction(self, valid_loader, task_id):
        self.net.eval()
        self.clear_all_list()
        pixel_auroc, img_auroc =0, 0

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