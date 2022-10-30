import torch
from torch import nn
import numpy as np
import argparse
import torch.nn.functional as F
from arch_base.base import ModelBase
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from loss_function.deviation_loss import DeviationLoss
from loss_function.binaryfocal_loss import BinaryFocalLoss


__all__ = ['DevNet']

def build_criterion(criterion):
    if criterion == 'deviation':
        return DeviationLoss()
    elif criterion == 'BCE':
        return torch.nn.BCEWithLogitsLoss()
    elif criterion == 'focal':
        return BinaryFocalLoss()
    elif criterion == 'CE':
        return torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

class _DevNet(nn.Module):
    def __init__(self, args, net):
        super(_DevNet, self).__init__()
        self.args = args
        self.net = net

        self.conv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, padding=0)

    def forward(self, image):
        if self.args._n_scales == 0:
            raise ValueError

        image_pyramid = list()
        for s in range(self.args._n_scales):
            image_scaled = F.interpolate(image, size=self.args._img_size // (2 ** s)) if s > 0 else image
            feature = self.net(image_scaled)

            scores = self.conv(feature)
            if self.args._topk > 0:
                scores = scores.view(int(scores.size(0)), -1)
                topk = max(int(scores.size(1) * self.args._topk), 1)
                scores = torch.topk(torch.abs(scores), topk, dim=1)[0]
                scores = torch.mean(scores, dim=1).view(-1, 1)
            else:
                scores = scores.view(int(scores.size(0)), -1)
                scores = torch.mean(scores, dim=1).view(-1, 1)

            image_pyramid.append(scores)
        scores = torch.cat(image_pyramid, dim=1)
        score = torch.mean(scores, dim=1)
        return score.view(-1, 1)


class DevNet(ModelBase):
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        super(DevNet, self).__init__(config, device, file_path, net, optimizer, scheduler)
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net

        self.args = argparse.Namespace(**self.config)
        self.model = _DevNet(self.args, self.net).to(self.device)
        
        self.criterion = build_criterion(self.args._criterion) 
        
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_model(self, train_loader, task_id, inf=''):
        self.model.train()
        self.scheduler.step()

        train_loss = 0.

        for epoch in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                image = batch['img'].to(self.device)
                target = batch['label'].to(self.device)

                output = self.model(image)
                loss = self.criterion(output, target.unsqueeze(1).float())
                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()

    def prediction(self, valid_loader, task_id):
        self.pixel_gt_list.clear()
        self.img_gt_list.clear()
        self.pixel_pred_list.clear()
        self.img_pred_list.clear()
        self.img_path_list.clear()
        self.model.eval()
        pixel_auroc, img_auroc = 0, 0

        test_loss = 0.0
        total_pred = np.array([])
        total_target = np.array([])

        for batch_id, batch in enumerate(valid_loader):
            image = batch['img'].to(self.device)
            target = batch['label'].to(self.device)

            with torch.no_grad():
                output = self.model(image.float())
            loss = self.criterion(output, target.unsqueeze(1).float())
            test_loss += loss.item()
            
            self.img_gt_list.append(target.cpu().numpy()[0])
            self.img_pred_list.append(output.data.cpu().numpy()[0])
            self.img_path_list.append(batch['img_src'])

        #     total_pred = np.append(total_pred, output.data.cpu().numpy())
        #     total_target = np.append(total_target, target.cpu().numpy())


        # img_auroc = roc_auc_score(total_target, total_pred)
        # ap = average_precision_score(total_target, total_pred)

        # return pixel_auroc, img_auroc