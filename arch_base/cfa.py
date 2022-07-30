from math import gamma
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

    def __init__(self, config, train_loaders, valid_loaders, device, 
                 file_path, train_fewshot_loaders=None):
    
        self.config = config
        self.train_loaders = train_loaders
        self.valid_loaders = valid_loaders
        self.device = device
        self.file_path = file_path
        self.train_fewshot_loaders = train_fewshot_loaders

        self.chosen_train_loaders = [] 
        if self.config['chosen_train_task_ids'] is not None:
            for idx in range(len(self.config['chosen_train_task_ids'])):
                self.chosen_train_loaders.append(self.train_loaders[self.config['chosen_train_task_ids'][idx]])
        else:
            self.chosen_train_loaders = self.train_loaders

        self.chosen_valid_loader = self.valid_loaders[self.config['chosen_test_task_id']] 

        if self.config['fewshot']:
            assert self.train_fewshot_loaders is not None
            self.chosen_fewshot_loader = self.train_fewshot_loaders[self.config['chosen_test_task_id']]
        
        if self.config['chosen_test_task_id'] in self.config['chosen_train_task_ids']:
            assert self.config['fewshot'] is False, 'Changeover: test task id should not be the same as train task id'
        
        if self.config['backbone'] == 'resnet18':
            self.backbone = resnet18(pretrained=True, progress=True)
        elif self.config['backbone'] == 'efficientnet':
            self.backbone = effnet(pretrained=True, progress=True)
        elif self.config['backbone'] == 'wide_resnet50':
            self.backbone = wide_resnet50_2(pretrained=True, progress=True)
        elif self.config['backbone'] == 'vgg':
            self.backbone = vgg19(pretrained=True, progress=True)

        self.loss_fn = DSVDD(model=self.backbone, data_loader=self.chosen_train_loaders,
                        cnn=self.config['backbone'], gamma_c=self.config['gamma_c'],
                        gamma_d=self.config['gamma_d'], device=self.device)

        self.loss_fn = self.loss_fn.to(self.device)

        self.pixel_gt_list = []
        self.img_gt_list = []
        self.pixel_pred_list = []
        self.img_pred_list = []
    
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
    def cal_img_roc(scores, gt_list):
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        #fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = CFA.roc_auc_img(gt_list, img_scores)

        #return fpr, tpr, img_roc_auc
        return img_roc_auc
        
        

    def train_on_epoch(self):

        self.backbone.eval()

        self.loss_fn.train()

        optimizer = torch.optim.AdamW(params=self.backbone.parameters(),
                                      lr=self.config['lr'],
                                      weight_decay=self.config['weight_decay'],
                                      amsgrad=True)

        # When num_task is 15, per task means per class
        for task_idx, train_loader in enumerate(self.chosen_train_loaders):
            print('run task: {}'.format(self.config['chosen_train_task_ids'][task_idx]))
            for epoch in range(self.config['num_epochs']): 
                for batch_id, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    img = batch['img'].to(self.device)
                    p = self.backbone(img)

                    loss, _ = self.loss_fn(p)
                    loss.backward()
                    optimizer.step()

    def prediction(self):
        self.loss_fn.eval()
        self.pixel_gt_list.clear()
        self.img_gt_list.clear()
        self.pixel_pred_list.clear()
        self.img_pred_list.clear()

        with torch.no_grad():
            for batch_id, batch in enumerate(self.chosen_valid_loader):

                img = batch['img'].to(self.device)
                label = batch['label'].to(self.device)
                mask = batch['mask'].to(self.device)

                self.img_gt_list.append(label.cpu().detach().numpy())
                self.pixel_gt_list.append(mask.cpu().detach().numpy())

                p = self.backbone(img)

                _, score = self.loss_fn(p)
                heatmap = score.cpu().detach()
                heatmap = torch.mean(heatmap, dim=1) 
                heatmaps = torch.cat((heatmaps, heatmap), dim=0) if heatmaps != None else heatmap
       
        heatmaps = CFA.upsample(heatmaps, size=img.size(2)) 
        heatmaps = CFA.gaussian_smooth(heatmaps, sigma=4)
        
        gt_mask = np.asarray(self.pixel_gt_list)
        scores = CFA.rescale(heatmaps)