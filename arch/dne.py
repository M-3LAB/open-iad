import torch
from torch import nn
import numpy as np
import argparse
import torch.nn.functional as F
from arch.base import ModelBase
from models.cutpaste.density import GaussianDensityTorch
from models.vit.vit import ViT
from optimizer.optimizer import get_optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

__all__ = ['DNE']

class _DNE(nn.Module):
    def __init__(self, args, net, optimizer, scheduler):
        super(_DNE, self).__init__()
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net
        self.softmax = nn.Softmax(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, epoch, inputs, labels, one_epoch_embeds, task_wise_mean, task_wise_cov, t):
        if self.args._strong_augmentation:
            num = int(len(inputs) / 2)
        else:
            num = int(len(inputs))

        if self.args._fix_head:
            if t >= 1:
                for param in self.net.head.parameters():
                    param.requires_grad = False

        self.optimizer.zero_grad()
        embeds, outs = self.net(inputs)
        one_epoch_embeds.append(embeds[:num].detach().cpu())
        loss = self.cross_entropy(self.softmax(outs), labels.long())
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step(epoch)

    def training_epoch(self, density, one_epoch_embeds, task_wise_mean, task_wise_cov, task_wise_train_data_nums, t):
        one_epoch_embeds = torch.cat(one_epoch_embeds)
        one_epoch_embeds = F.normalize(one_epoch_embeds, p=2, dim=1)
        mean, cov = density.fit(one_epoch_embeds)

        if len(task_wise_mean) < t + 1:
            task_wise_mean.append(mean)
            task_wise_cov.append(cov)
        else:
            task_wise_mean[-1] = mean
            task_wise_cov[-1] = cov

        task_wise_embeds = []
        for i in range(t + 1):
            if i < t:
                past_mean, past_cov, past_nums = task_wise_mean[i], task_wise_cov[i], task_wise_train_data_nums[i]
                past_embeds = np.random.multivariate_normal(past_mean, past_cov, size=past_nums)
                task_wise_embeds.append(torch.FloatTensor(past_embeds))
            else:
                task_wise_embeds.append(one_epoch_embeds)
        for_eval_embeds = torch.cat(task_wise_embeds, dim=0)
        for_eval_embeds = F.normalize(for_eval_embeds, p=2, dim=1)
        _, _ = density.fit(for_eval_embeds)
        
        return density

class DNE(ModelBase):
    def __init__(self, config):
        super(DNE, self).__init__(config)
        self.config = config

        self.net = ViT(num_classes=self.config['_num_classes'], pretrained=self.config['_pretrained'], checkpoint_path='./checkpoints/vit/vit_b_16.npz')
        self.optimizer = get_optimizer(self.config, self.net.parameters())
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, self.config['num_epochs'])
        self.args = argparse.Namespace(**self.config)
        self.model = _DNE(self.args, self.net, self.optimizer, self.scheduler).to(self.device)

        self.density = GaussianDensityTorch()
        self.one_epoch_embeds = []
        self.task_wise_mean = []
        self.task_wise_cov = []
        self.task_wise_train_data_nums = []

    def train_model(self, train_loader, task_id, inf=''):
        self.net.train()

        for epoch in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                inputs = batch['img'].to(self.device)
                labels = batch['label'].to(self.device)

                self.model(epoch, inputs, labels, self.one_epoch_embeds, self.task_wise_mean, self.task_wise_cov, task_id)

        self.task_wise_train_data_nums.append(len(train_loader) * self.config['train_batch_size'])

    def prediction(self, valid_loader, task_id):
        self.model.eval()
        self.clear_all_list()

        density = self.model.training_epoch(self.density, self.one_epoch_embeds, self.task_wise_mean, 
                                          self.task_wise_cov, self.task_wise_train_data_nums, task_id)
        if self.args._eval_classifier == 'density':
            labels, embeds = [], []
            with torch.no_grad():
                for batch_id, batch in enumerate(valid_loader):
                    x = batch['img'].to(self.device)
                    label = batch['label'].to(self.device)
                    embed = self.net.forward_features(x)
                    embeds.append(embed.cpu())
                    labels.append(label.cpu())
                    self.img_path_list.append(batch['img_src'])

            labels = torch.cat(labels)
            embeds = torch.cat(embeds)
            embeds = F.normalize(embeds, p=2, dim=1)
            distances = density.predict(embeds)
            self.img_gt_list = labels
            self.img_pred_list = distances

        elif self.args._eval_classifier == 'head':
            with torch.no_grad():
                for batch_id, batch in enumerate(valid_loader):
                    x = batch['img'].to(self.device)
                    label = batch['label'].to(self.device)
                    _, out = self.net(x)
                    _, out = torch.max(out, 1)
                    self.img_pred_list.append(out.cpu().numpy()[0])
                    self.img_gt_list.append(label.cpu().numpy()[0])
                    self.img_path_list.append(batch['img_src'])