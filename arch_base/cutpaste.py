import torch
import torch.nn as nn
import torch.nn.functional as F

from arch_base.base import ModelBase
from tools.density import GaussianDensityTorch


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
        num = int(len(inputs) / 2)
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
        super(CutPaste, self).__init__(config, device, file_path, net, optimizer, scheduler)
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = _CutPaste(self.config, self.net, optimizer, scheduler).to(self.device)
        self.density = GaussianDensityTorch()
        self.one_epoch_embeds = []
    
    def train_model(self, train_loader, task_id, inf=''):
        for epoch in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                self.net.train()
                imgs = batch['img']
                inputs = [img.to(self.device) for img in imgs]
                labels = torch.arange(len(inputs), device=self.device)
                labels = labels.repeat_interleave(inputs[0].size(0))
                inputs = torch.cat(inputs, dim=0)
                self.model(epoch, inputs, labels, self.one_epoch_embeds)
        
    def prediction(self, valid_loader, task_id=None):
        self.net.eval()
        density = self.model.training_epoch(self.density, self.one_epoch_embeds)
        labels = []
        embeds = []
        
        with torch.no_grad():
            for batch_id, batch in enumerate(valid_loader):
                input = batch['img'].to(self.device)
                label = batch['label'].to(self.device)
                self.img_path_list.append(batch['img_src'])

                embed = self.net.forward_features(input)
                embeds.append(embed.cpu())
                labels.append(label.cpu())
            
            labels = torch.cat(labels)
            embeds = torch.cat(embeds)
            embeds = F.normalize(embeds, p=2, dim=1)

            distances = density.predict(embeds)
            self.img_gt_list = labels
            self.img_pred_list = distances
