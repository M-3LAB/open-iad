import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from arch.base import ModelBase
from models.favae.net_favae import NetFAVAE
from torchvision import models
from models.favae.func import EarlyStop, AverageMeter
from scipy.ndimage import gaussian_filter
from optimizer.optimizer import get_optimizer


__all__ = ['FAVAE']

class FAVAE(ModelBase):
    def __init__(self, config):
        super(FAVAE, self).__init__(config)
        self.config = config

        self.vaenet = NetFAVAE().to(self.device)
        self.optimizer = get_optimizer(self.config, self.vaenet.parameters())
        self.scheduler = None
        self.teacher = models.vgg16(pretrained=True).to(self.device)
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.early_stop = EarlyStop(patience=20, save_name='favae.pt')
        self.criterion_1 = nn.MSELoss(reduction='sum')
        self.criterion_2 = nn.MSELoss(reduction='none')

    def feature_extractor(self, x, model, target_layers):
        target_activations = list()
        for name, module in model._modules.items():
            x = module(x)
            if name in target_layers:
                target_activations += [x]
        return target_activations, x

    def train_model(self, train_loader, task_id, inf=''):
        self.vaenet.train()
        self.teacher.eval()

        losses = AverageMeter()
        for epoch in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                img = batch['img'].to(self.device)
                z, output, mu, log_var = self.vaenet(img)
                s_activations, _ = self.feature_extractor(z, self.vaenet.decode, target_layers=['10', '16', '22'])
                t_activations, _ = self.feature_extractor(img, self.teacher.features, target_layers=['7', '14', '21'])

                self.optimizer.zero_grad()
                mse_loss = self.criterion_1(output, img)
                kld_loss = 0.5 * torch.sum(-1 - log_var + torch.exp(log_var) + mu**2)
                for i in range(len(s_activations)):
                    s_act = self.vaenet.adapter[i](s_activations[-(i + 1)])
                    mse_loss += self.criterion_1(s_act, t_activations[i])
                loss = mse_loss + self.config['_kld_weight'] * kld_loss
                losses.update(loss.sum().item(), img.size(0))

                loss.backward()
                self.optimizer.step()

    def prediction(self, valid_loader, task_id=None):
        self.vaenet.eval()
        self.teacher.eval()
        self.clear_all_list()

        pixel_pred_list = []
        gt_mask_list = []
        recon_imgs = []

        with torch.no_grad():
            for batch_id, batch in enumerate(valid_loader):
                img = batch['img'].to(self.device)
                mask = batch['mask'].numpy()
                label = batch['label']
                z, output, mu, log_var = self.vaenet(img)
                s_activations, _ = self.feature_extractor(z, self.vaenet.decode, target_layers=['10', '16', '22']) 
                t_activations, _ = self.feature_extractor(img, self.teacher.features, target_layers=['7', '14', '21'])

                score = self.criterion_2(output, img).sum(1, keepdim=True)

                for i in range(len(s_activations)):
                    s_act = self.vaenet.adapter[i](s_activations[-(i + 1)])
                    mse_loss = self.criterion_2(s_act, t_activations[i]).sum(1, keepdim=True)
                    score += F.interpolate(mse_loss, size=img.size(2), mode='bilinear', align_corners=False)
                
                score = score.squeeze().cpu().numpy()

                for i in range(score.shape[0]):
                    score[i] = gaussian_filter(score[i], sigma=4)
                pixel_pred_list.append(score.reshape(img.size(2),img.size(2)))
                recon_imgs.extend(output.cpu().numpy())
                mask[mask >= 0.5] = 1
                mask[mask < 0.5] = 0
                gt_mask_list.append(mask[0, 0].astype(int))
                self.img_gt_list.append(label.numpy()[0])
                self.img_pred_list.append(np.max(score))
                self.img_path_list.append(batch['img_src'])
        
        max_anomaly_score = np.array(pixel_pred_list).max()
        min_anomaly_score = np.array(pixel_pred_list).min()
        pixel_pred_list = (pixel_pred_list - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
        self.pixel_gt_list = gt_mask_list
        self.pixel_pred_list = pixel_pred_list