import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.rd_resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from utils.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50



class NetRevDis(nn.Module):
    def __init__(self, args):
        super(NetRevDis, self).__init__()
        self.args = args
        self.encoder, self.bn = wide_resnet50_2(pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.decoder = de_wide_resnet50_2(pretrained=False)

    def forward(self, imgs):
        inputs = self.encoder(imgs)
        outputs = self.decoder(self.bn(inputs))
        return inputs, outputs


class RevDis(nn.Module):
    def __init__(self, args, net, optimizer, scheduler):
        super(RevDis, self).__init__()
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net

    def loss_fucntion(self, a, b):
        cos_loss = torch.nn.CosineSimilarity()
        loss = 0
        for item in range(len(a)):
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1)))
        return loss

    def forward(self, epoch, inputs, labels, one_epoch_embeds, *args):
        self.optimizer.zero_grad()
        t_outs, outs = self.net(inputs)
        loss = self.loss_fucntion(t_outs, outs)
        loss.backward()
        self.optimizer.step()


    def training_epoch(self, density, one_epoch_embeds, *args):
        pass

