import torch
from torch import nn
import numpy as np
import argparse
import torch.nn.functional as F
from arch_base.base import ModelBase
from tools.density import GaussianDensityTorch
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve

__all__ = ['DNE']

# contastive svdd
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
        for i in range(t+1):
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
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        super(DNE, self).__init__(config, device, file_path, net, optimizer, scheduler)
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net

        self.args = argparse.Namespace(**self.config)
        self.model = _DNE(self.args, self.net, optimizer, scheduler).to(self.device)
        
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


    def prediction(self, valid_loader, task_id):
        self.pixel_gt_list.clear()
        self.img_gt_list.clear()
        self.pixel_pred_list.clear()
        self.img_pred_list.clear()
        self.img_path_list.clear()
        self.model.eval()
        pixel_auroc, img_auroc = 0, 0

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
            # roc_auc = roc_auc_score(labels, distances)
            # fpr, tpr, _ = roc_curve(labels, distances)
            # img_auroc = auc(fpr, tpr)

        elif self.args._eval_classifier == 'head':
            # labels, outs = [], []
            with torch.no_grad():
                for batch_id, batch in enumerate(valid_loader):
                    x = batch['img'].to(self.device)
                    label = batch['label'].to(self.device)
                    _, out = self.net(x)
                    _, out = torch.max(out, 1)
                    self.img_pred_list.append(out.cpu().numpy()[0])
                    self.img_gt_list.append(label.cpu().numpy()[0])
                    self.img_path_list.append(batch['img_src'])
        #             outs.append(out.cpu())
        #             labels.append(label.cpu())

        #     labels = torch.cat(labels)
        #     outs = torch.cat(outs)
        #     img_auroc = roc_auc_score(labels, outs)

        # return pixel_auroc, img_auroc