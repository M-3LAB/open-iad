import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CompactnessLoss(nn.Module):
    def __init__(self, center):
        super(CompactnessLoss, self).__init__()
        self.center = center

    def forward(self, inputs):
        m = inputs.size(1)
        variances = (inputs - self.center).norm(dim=1).pow(2) / m
        return variances.mean()

# contastive svdd
class Dis(nn.Module):
    def __init__(self, args, net, optimizer, scheduler):
        super(Dis, self).__init__()
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net
        self.softmax = nn.Softmax(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, epoch, inputs, labels, one_epoch_embeds, task_wise_mean, task_wise_cov, t):
        if self.args.dataset.strong_augmentation:
            num = int(len(inputs) / 2)
        else:
            num = int(len(inputs))

        if self.args.model.fix_head:
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


