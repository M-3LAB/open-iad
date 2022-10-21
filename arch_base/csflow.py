import torch
import torch.nn as nn
from torchvision import models
from models.patchcore.kcenter_greedy import KCenterGreedy 
from torchvision import transforms
import cv2
from typing import List
from tools.utilize import *
import os
import torch.nn.functional as F
import numpy as np
from sklearn.random_projection import SparseRandomProjection
import faiss
#import tqdm
import math
from scipy.ndimage import gaussian_filter
from metrics.common.np_auc_precision_recall import np_get_auroc
from tools.visualize import save_anomaly_map, vis_embeddings
from memory_augmentation.domain_generalization import feature_augmentation

__all__ = ['PatchCore2D']

class CSFlow():
    def __init__(self, config, device, file_path):
        
        self.config = config
        self.device = device
        self.file_path = file_path

        # Backbone model
        if config['backbone'] == 'resnet18':
            self.backbone = models.resnet18(pretrained=True, progress=True).to(self.device)
        elif config['backbone'] == 'wide_resnet50':
            self.backbone = models.wide_resnet50_2(pretrained=True, progress=True).to(self.device)
        else:
            raise NotImplementedError('This Pretrained Model Not Implemented Error')

        self.features = [] 

        self.pixel_gt_list = []
        self.img_gt_list = []
        self.pixel_pred_list = []
        self.img_pred_list = []

        self.embeddings_list = []

        source_domain = ''
        if self.config['continual']:
            for i in self.config['train_task_id']:  
                source_domain = source_domain + str(self.config['train_task_id'][i])
        else:
            source_domain = str(self.config['train_task_id'][0])

    

        
    def train_epoch(self, train_loaders, inf=''):
        # for vanilla, fewshot, noisy

        self.backbone.eval()
        # When num_task is 15, per task means per class
        for task_idx, train_loader in enumerate(train_loaders):
            print('run task: {}'.format(self.config['train_task_id'][task_idx]))
            for _ in range(self.config['num_epochs']):
                for batch_id, batch in enumerate(train_loader):
                    # print(f'batch id: {batch_id}')
                    #if self.config['debug'] and batch_id > self.config['batch_limit']:
                    #    break
                    img = batch['img'].to(self.device)
                    #mask = batch['mask'].to(self.device)

                    # Extract features from backbone
                    self.features.clear()
                    _ = self.backbone(img)


    def prediction(self, valid_loader):

        self.backbone.eval()



        return pixel_auroc, img_auroc



        
      

