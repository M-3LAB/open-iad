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
class Discat(nn.Module):
    def __init__(self, args, net, optimizer, scheduler):
        super(Discat, self).__init__()
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

        self.optimizer.zero_grad()
        if self.args.model.save_anormal:
            out, embeds = self.net(inputs)
            noaug_embeds, aug_embeds = embeds[:num], embeds[num:]
            loss = self.cross_entropy(self.softmax(out), labels)  # + 0.5 * F.cosine_similarity(noaug_embeds, aug_embeds, dim=-1).mean()
            batch_embeds = [noaug_embeds.detach().cpu(), aug_embeds.detach().cpu()]
            one_epoch_embeds.append(batch_embeds)

            for i in range(len(task_wise_mean)-1):
                idx = -(i+2)
                noaug_past_embeds = np.random.multivariate_normal(task_wise_mean[idx][0], task_wise_cov[idx][0], size=num)
                noaug_past_embeds = torch.cuda.FloatTensor(noaug_past_embeds)
                noaug_past_out = self.net.head(noaug_past_embeds)
                noaug_past_labels = torch.zeros(len(inputs), device=self.args.device).long()
                loss += self.cross_entropy(self.softmax(noaug_past_out), noaug_past_labels)

                aug_past_embeds = np.random.multivariate_normal(task_wise_mean[idx][1], task_wise_cov[idx][1], size=num)
                aug_past_embeds = torch.cuda.FloatTensor(aug_past_embeds)
                aug_past_out = self.net.head(aug_past_embeds)
                aug_past_labels = torch.ones(len(inputs), device=self.args.device).long()
                loss += self.cross_entropy(self.softmax(aug_past_out), aug_past_labels)
        else:
            embeds, outs = self.net(inputs)
            one_epoch_embeds.append(embeds[:num].detach().cpu())
            loss = self.cross_entropy(self.softmax(outs), labels.long())
            for i in range(len(task_wise_mean) - 1):
                idx = -(i + 2)
                past_embeds = np.random.multivariate_normal(task_wise_mean[idx], task_wise_cov[idx],
                                                            size=len(inputs))
                past_embeds = torch.cuda.FloatTensor(past_embeds)
                past_out = self.net.head(past_embeds)
                past_labels = torch.zeros(len(inputs), device=self.args.device).long()
                loss += 0.4 * self.cross_entropy(self.softmax(past_out), past_labels)

        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step(epoch)

    def training_epoch(self, density, one_epoch_embeds, task_wise_mean, task_wise_cov, task_wise_train_data_nums, t):
        if self.args.model.save_anormal:
            noaug_one_epoch_embeds = [i[0] for i in one_epoch_embeds]
            noaug_one_epoch_embeds = torch.cat(noaug_one_epoch_embeds)
            noaug_one_epoch_embeds = F.normalize(noaug_one_epoch_embeds, p=2, dim=1)
            noaug_mean, noaug_cov = density.fit(noaug_one_epoch_embeds)

            aug_one_epoch_embeds = [i[1] for i in one_epoch_embeds]
            aug_one_epoch_embeds = torch.cat(aug_one_epoch_embeds)
            aug_one_epoch_embeds = F.normalize(aug_one_epoch_embeds, p=2, dim=1)
            aug_mean, aug_cov = density.fit(aug_one_epoch_embeds)

            mean = [noaug_mean, aug_mean]
            cov = [noaug_cov, aug_cov]

            if len(task_wise_mean) < t + 1:
                task_wise_mean.append(mean)
                task_wise_cov.append(cov)
            else:
                task_wise_mean[-1] = mean
                task_wise_cov[-1] = cov

            # task_wise_embeds = []
            # for i in range(t+1):
            #     if i < t:
            #         past_mean, past_cov, past_nums = task_wise_mean[i][0], task_wise_cov[i][0], task_wise_train_data_nums[i]
            #         past_embeds = np.random.multivariate_normal(past_mean, past_cov, size=past_nums)
            #         task_wise_embeds.append(torch.FloatTensor(past_embeds))
            #     else:
            #         task_wise_embeds.append(noaug_one_epoch_embeds)
            # for_eval_embeds = torch.cat(task_wise_embeds, dim=0)
            # for_eval_embeds = F.normalize(for_eval_embeds, p=2, dim=1)
            # _, _ = density.fit(for_eval_embeds)

        else:
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
        return density, task_wise_mean, task_wise_cov

