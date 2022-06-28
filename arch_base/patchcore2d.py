from re import I
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

__all__ = ['PatchCore2D']

class PatchCore2D():
    def __init__(self, config, train_loaders, valid_loaders, device, file_path):
        
        self.config = config
        self.train_loaders = train_loaders
        self.valid_loaders = valid_loaders
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
        self.get_layer_features(features=self.features)

        self.pixel_gt_list = []
        self.img_gt_list = []
        self.pixel_pred_list = []
        self.img_pred_list = []

        self.embeddings_list = []

    def get_layer_features(self, features: List):

        def hook_t(module, input, output):
            features.append(output)
        
        self.backbone.layer2[-1].register_forward_hook(hook_t)
        self.backbone.layer3[-1].register_forward_hook(hook_t)

    @staticmethod 
    def torch_to_cv(torch_img):
        inverse_normalization = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], 
                                                     std=[1/0.229, 1/0.224, 1/0.255])
        torch_img = inverse_normalization(torch_img)
        cv_img = cv2.cvtColor(torch_img.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB) 
        return cv_img

    @staticmethod
    def embedding_concate(x, y):
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z
        
    def train_epoch(self, inf=''):
        
        self.backbone.eval()

        for epoch in range(self.config['num_epoch']):
            for task_idx, train_loader in enumerate(self.train_loaders):

                print('run task: {}'.format(task_idx))
                create_folders(os.path.join(self.file_path, 'embeddings', str(task_idx)))
                create_folders(os.path.join(self.file_path, 'samples', str(task_idx)))
                self.embeddings_list.clear()

                for batch_id, batch in enumerate(train_loader):
                    if self.config['debug'] and batch_id > self.batch_limit:
                        break
                    img = batch['img'].to(self.device)
                    #mask = batch['mask'].to(self.device)

                    # Extract features from backbone
                    self.features.clear()
                    _ = self.backbone(img)

                    # Pooling for layer 2 and layer 3 features
                    embeddings = []
                    for feat in self.features:
                        pooling = torch.nn.AvgPool2d(3, 1, 1)
                        embeddings.append(pooling(feat))

                    embedding = PatchCore2D.embedding_concate(embeddings[0], embeddings[1])
                    print(f'embedding.size: {embedding.size()}')
                    print(f'embeddings[0].size: {embeddings[0].size()}')
                    print(f'embeddings[1].size: {embeddings[1].size()}')


                              
                    
                    
    def prediction(self):
        pass
      

