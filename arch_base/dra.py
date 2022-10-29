import torch
from torch import nn
import numpy as np
import argparse
import torch.nn.functional as F
from arch_base.base import ModelBase
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from models.dra.dra_resnet18 import *
from loss_function.deviation_loss import DeviationLoss
from loss_function.binaryfocal_loss import BinaryFocalLoss


__all__ = ['DRA']

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

class _DRA(nn.Module):
    def __init__(self, args, net):
        super(_DRA, self).__init__()
        self.args = args
        self.net = net

        self.in_c = 512
        self.holistic_head = HolisticHead(self.in_c)
        self.seen_head = PlainHead(self.in_c, self.args._topk)
        self.pseudo_head = PlainHead(self.in_c, self.args._topk)
        self.composite_head = CompositeHead(self.in_c, self.args._topk)

    def forward(self, image, label):
        image_pyramid = list()
        for i in range(self.args._total_heads):
            image_pyramid.append(list())
        for s in range(self.args._n_scales):
            image_scaled = F.interpolate(image, size=self.args._img_size // (2 ** s)) if s > 0 else image
            feature = self.net(image_scaled)

            ref_feature = feature[:self.args._nRef, :, :, :]
            feature = feature[self.args._nRef:, :, :, :]

            if self.training:
                normal_scores = self.holistic_head(feature)
                abnormal_scores = self.seen_head(feature[label != 2])
                dummy_scores = self.pseudo_head(feature[label != 1])
                comparison_scores = self.composite_head(feature, ref_feature)
            else:
                normal_scores = self.holistic_head(feature)
                abnormal_scores = self.seen_head(feature)
                dummy_scores = self.pseudo_head(feature)
                comparison_scores = self.composite_head(feature, ref_feature)
            for i, scores in enumerate([normal_scores, abnormal_scores, dummy_scores, comparison_scores]):
                image_pyramid[i].append(scores)
        for i in range(self.args._total_heads):
            image_pyramid[i] = torch.cat(image_pyramid[i], dim=1)
            image_pyramid[i] = torch.mean(image_pyramid[i], dim=1)

        return image_pyramid


class DRA(ModelBase):
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net

        self.args = argparse.Namespace(**self.config)
        self.model = _DRA(self.args, self.net).to(self.device)
        
        self.criterion = build_criterion(self.args._criterion) 
        
        self.optimizer = optimizer
        self.scheduler = scheduler

    def generate_target(self, target, eval=False):
        targets = []
        if eval:
            targets.append(target==0)
            targets.append(target)
            targets.append(target)
            targets.append(target)
            return targets
        else:
            temp_t = target != 0
            targets.append(target == 0)
            targets.append(temp_t[target != 2])
            targets.append(temp_t[target != 1])
            targets.append(target != 0)
        return targets

    def normalization(self, data):
        return data

    def train_model(self, train_ref_loader, task_id, inf=''):
        train_loader, refer_loader = train_ref_loader
        ref = iter(refer_loader)
        self.model.train()
        self.scheduler.step()

        train_loss = 0.
        class_loss = [0. for j in range(self.args._total_heads)]

        for epoch in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                image = batch['img'].to(self.device)
                target = batch['label'].to(self.device)

                if self.args._total_heads == 4:
                    try:
                        ref_image = next(ref)['img']
                    except StopIteration:
                        ref = iter(refer_loader)
                        ref_image = next(ref)['img']
                    ref_image = ref_image.to(self.device)
                    image = torch.cat([ref_image, image], dim=0)

                outputs = self.model(image, target)
                targets = self.generate_target(target)

                losses = []
                for i in range(self.args._total_heads):
                    if self.args._criterion == 'CE':
                        prob = F.softmax(outputs[i], dim=1)
                        losses.append(self.criterion(prob, targets[i].long()).view(-1, 1))
                    else:
                        losses.append(self.criterion(outputs[i], targets[i].float()).view(-1, 1))

                loss = torch.cat(losses)
                loss = torch.sum(loss)

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()
                for i in range(self.args._total_heads):
                    class_loss[i] += losses[i].item()

    def prediction(self, valid_ref_loader, task_id):
        valid_loader, ref_loader = valid_ref_loader
        ref = iter(ref_loader)
        self.model.eval()
        pixel_auroc, img_auroc = 0, 0

        test_loss = 0.0
        class_pred = [np.array([]) for i in range(self.args._total_heads)]
        total_target = np.array([])

        for batch_id, batch in enumerate(valid_loader):
            image = batch['img'].to(self.device)
            target = batch['label'].to(self.device)

            if self.args._total_heads == 4:
                try:
                    ref_image = next(ref)['img']
                except StopIteration:
                    ref = iter(ref_loader)
                    ref_image = next(ref)['img']
                ref_image = ref_image.to(self.device)
                image = torch.cat([ref_image, image], dim=0)

            with torch.no_grad():
                outputs = self.model(image, target)
                targets = self.generate_target(target, eval=True)

                losses = list()
                for i in range(self.args._total_heads):
                    if self.args._criterion == 'CE':
                        prob = F.softmax(outputs[i], dim=1)
                        losses.append(self.criterion(prob, targets[i].long()))
                    else:
                        losses.append(self.criterion(outputs[i], targets[i].float()))

                loss = losses[0]
                for i in range(1, self.args._total_heads):
                    loss += losses[i]

            test_loss += loss.item()

            for i in range(self.args._total_heads):
                if i == 0:
                    data = -1 * outputs[i].data.cpu().numpy()
                else:
                    data = outputs[i].data.cpu().numpy()
                class_pred[i] = np.append(class_pred[i], data)
            total_target = np.append(total_target, target.cpu().numpy())

        total_pred = self.normalization(class_pred[0])
        for i in range(1, self.args._total_heads):
            total_pred = total_pred + self.normalization(class_pred[i])


        img_auroc = roc_auc_score(total_target, total_pred)
        ap = average_precision_score(total_target, total_pred)

        return pixel_auroc, img_auroc