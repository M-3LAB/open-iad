from math import gamma
from models.cfa.metrics import get_threshold, cal_img_roc, cal_pxl_roc, upsample, rescale, gaussian_smooth
import torch
import torch.nn as nn
from models.cfa.efficientnet import EfficientNet as effnet
from models.cfa.resnet import wide_resnet50_2, resnet18
from models.cfa.vgg import vgg19_bn as vgg19
from models.cfa.cfa import DSVDD
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import numpy as np
from sklearn.metrics import roc_auc_score

__all__ = ['CFA']

class CFA():

    def __init__(self, config, device, file_path, net, optimizer, scheduler):
    
        self.config = config
        self.device = device
        self.file_path = file_path
        #self.backbone = net
        #self.optimizer = optimizer

        if self.config['backbone'] == 'resnet18':
            self.backbone = resnet18(pretrained=True, progress=True)
        #elif self.config['backbone'] == 'efficientnet':
        #    self.backbone = effnet(pretrained=True, progress=True)
        #elif self.config['backbone'] == 'wide_resnet50':
        #    self.backbone = wide_resnet50_2(pretrained=True, progress=True)
        #elif self.config['backbone'] == 'vgg':
        #    self.backbone = vgg19(pretrained=True, progress=True)

        self.backbone.to(self.device)

        
        self.pixel_gt_list = []
        self.img_gt_list = []
        self.pixel_pred_list = []
        self.img_pred_list = []

        self.best_img_auroc = -1
        self.best_pixel_auroc = -1
    
    @staticmethod 
    def upsample(x, size, mode):
        return F.interpolate(x.unsqueeze(1), size=size, mode=mode, align_corners=False).squeeze().numpy()
    
    @staticmethod
    def gaussian_smooth(x, sigma=4):
        bs = x.shape[0]
        for i in range(0, bs):
            x[i] = gaussian_filter(x[i], sigma=sigma)

        return x
    
    @staticmethod
    def rescale(x):
        return (x - x.min()) / (x.max() - x.min())
    
    @staticmethod
    def roc_auc_img(gt, score):
        img_roc_auc = roc_auc_score(gt, score)
        return img_roc_auc
    
    @staticmethod
    def roc_auc_pixel(gt, score):
        per_pixel_roc_auc = roc_auc_score(gt.flatten(), score.flatten())
        return per_pixel_roc_auc
        
    
    @staticmethod
    def cal_img_roc(scores, gt_list):
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        #fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = CFA.roc_auc_img(gt_list, img_scores)

        #return fpr, tpr, img_roc_auc
        return img_roc_auc
    
    def cal_pxl_roc(gt_mask, scores):
        #fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = CFA.roc_auc_pixel(gt_mask.flatten(), scores.flatten())
    
        #return fpr, tpr, per_pixel_rocauc
        return per_pixel_rocauc

    def train_model(self, train_loader, task_id, inf=''):

        self.loss_fn = DSVDD(model=self.backbone, data_loader=train_loader,
                             cnn=self.config['backbone'], gamma_c=self.config['gamma_c'],
                             gamma_d=self.config['gamma_d'], device=self.device)

        self.loss_fn = self.loss_fn.to(self.device)

        self.backbone.eval()

        self.loss_fn.train()

        params = [{'params' : self.loss_fn.parameters()},]
        optimizer = torch.optim.AdamW(params=params, lr=1e-3, weight_decay=5e-4,
                                      amsgrad=True)

        for epoch in range(self.config['num_epochs']): 
            for batch_id, batch in enumerate(train_loader):
                optimizer.zero_grad()
                img = batch['img'].to(self.device)
                p = self.backbone(img)

                loss, _ = self.loss_fn(p)
                loss.backward()
                optimizer.step()
            

    def prediction(self, valid_loader, task_id=None):
        self.loss_fn.eval()
        gt_mask_list = list()
        gt_list = list()
        heatmaps = None

        with torch.no_grad():
            for batch_id, batch in enumerate(valid_loader):

                img = batch['img'].to(self.device)
                label = batch['label']
                mask = batch['mask'].to(self.device)
                mask[mask>=0.5] = 1
                mask[mask<0.5] = 0

                gt_list.extend(label.cpu().detach().numpy())
                gt_mask_list.extend(mask.cpu().detach().numpy())

                p = self.backbone(img)

                _, score = self.loss_fn(p)
                heatmap = score.cpu().detach()
                heatmap = torch.mean(heatmap, dim=1) 
                heatmaps = torch.cat((heatmaps, heatmap), dim=0) if heatmaps != None else heatmap
       
        heatmaps = upsample(heatmaps, size=img.size(2), mode='bilinear') 
        heatmaps = gaussian_smooth(heatmaps, sigma=4)
        
        gt_mask = np.asarray(gt_mask_list)
        scores = rescale(heatmaps)
        
        fpr, tpr, img_auroc = cal_img_roc(scores, gt_list)
        pixel_auroc = cal_pxl_roc(gt_mask, scores)

        return pixel_auroc, img_auroc