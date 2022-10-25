import torch
import torch.nn as nn
import torch.nn.functional as F

from arch_base.base import ModelBase

__all__ = ['CutPaste']

class _CutPaste(nn.Module):
    def __init__(self, args, net, optimizer, scheduler):
        super(_CutPaste, self).__init__()
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net
        self.softmax = nn.Softmax(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, epoch, inputs, labels, one_epoch_embeds, *args):
        if self.args.dataset.strong_augmentation:
            num = int(len(inputs) / 2)
        else:
            num = int(len(inputs))

        self.optimizer.zero_grad()
        embeds, outs = self.net(inputs)
        one_epoch_embeds.append(embeds[:num].detach().cpu())
        loss = self.cross_entropy(self.softmax(outs), labels.long())
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step(epoch)

    def training_epoch(self, density, one_epoch_embeds, *args):
        one_epoch_embeds = torch.cat(one_epoch_embeds)
        one_epoch_embeds = F.normalize(one_epoch_embeds, p=2, dim=1)
        _, _ = density.fit(one_epoch_embeds)
        return density
class CutPaste(ModelBase):
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        super().__init__()
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def train_model(self, train_loaders, inf=''):
        pass

    def prediction(self, valid_loader, task_id=None):
        pass