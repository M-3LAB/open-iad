import torch
from torch import nn
import argparse
import torch.nn.functional as F
from arch.base import ModelBase
from loss_function.deviation import DeviationLoss
from loss_function.binaryfocal import BinaryFocalLoss
from models.devnet.devnet_resnet18 import DevNetResNet18
from optimizer.optimizer import get_optimizer

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
    def __init__(self, config):
        super(DevNet, self).__init__(config)
        self.config = config

        self.net = DevNetResNet18()
        self.optimizer = get_optimizer(self.config, self.net.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['_step_size'], gamma=self.config['_gamma'])  
        args = argparse.Namespace(**self.config)
        self.model = _DevNet(args, self.net).to(self.device)
        self.criterion = build_criterion(self.config['_criterion']) 

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
        self.model.eval()
        self.clear_all_list()
        test_loss = 0.0

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
