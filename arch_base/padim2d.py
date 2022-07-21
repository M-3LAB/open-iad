from tools.utilize import save_feat_pickle
import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from random import sample
import torch.nn.functional as F
import numpy as np
import os
from tools.utilize import *
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from metrics.common.np_auc_precision_recall import np_get_auroc

__all__ = ['PaDim']

class PaDim():
    def __init__(self, config, train_loaders, valid_loaders, device, file_path, train_fewshot_loaders=None):
        
        self.config = config
        self.train_loader = train_loaders
        self.valid_loader = valid_loaders
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
        
        # Backbone model
        if config['backbone'] == 'resnet18':
            self.backbone = models.resnet18(pretrained=True, progress=True).to(self.device)
            t_d = 448
            d = 100
        elif config['backbone'] == 'wide_resnet50':
            self.backbone = models.wide_resnet50_2(pretrained=True, progress=True).to(self.device)
            t_d = 1792
            d = 550
        else:
            raise NotImplementedError('This Pretrained Model Not Implemented Error')

        # random select d dimension 
        self.idx = torch.tensor(sample(range(0, t_d), d))

        self.features = []

        self.train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        self.test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])]) 

        for i in range(len(self.config['chosen_train_task_ids'])):
            if i == 0:
                source_domain = str(self.config['chosen_train_task_ids'][0])
            else:
                source_domain = source_domain + str(self.config['chosen_train_task_ids'][i])

        #target_domain = str(self.config['chosen_test_task_id'])
        self.embedding_dir_path = os.path.join(self.file_path, 'embeddings', 
                                          source_domain)
        create_folders(self.embedding_dir_path)

        self.pixel_gt_list = []
        self.img_gt_list = []
        #self.pixel_pred_list = []
        #self.img_pred_list = []

        #self.total_img_rocauc = []
        #self.total_pixel_rocauc = []

    def get_layer_features(self):
    
        def hook_t(module, input, output):
            self.features.append(output)
        
        self.backbone.layer1[-1].register_forward_hook(hook_t)
        self.backbone.layer2[-1].register_forward_hook(hook_t)
        self.backbone.layer3[-1].register_forward_hook(hook_t)
    
    @staticmethod
    def dict_clear(outputs):
        for key, value in outputs.items():
            if isinstance(value, list): 
                value.clear()
    
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
        # When num_task is 15, per task means per class
        self.get_layer_features(outputs=self.train_outputs)

        for task_idx, train_loader in enumerate(self.chosen_train_loaders):

            PaDim.dict_clear(self.train_outputs)

            print('run task: {}'.format(self.config['chosen_train_task_ids'][task_idx]))

            for _ in range(self.config['num_epoch']):
                for batch_id, batch in enumerate(train_loader):
                    #print(f'batch id: {batch_id}')
                    img = batch['img'].to(self.device) 

                    self.features.clear()
                    with torch.no_grad():
                        _ = self.backbone(img)
                    
                    #get the intermediate layer outputs
                    for k,v in zip(self.train_outputs.keys(), self.features):
                        self.train_outputs[k].append(v.cpu().detach())
                
            for k, v in self.train_outputs.items():
                self.train_outputs[k] = torch.cat(v, 0)
            
            embedding_vectors = self.train_outputs['layer1']
            for layer_name in ['layer2', 'layer3']:
                embedding_vectors = PaDim.embedding_concate(embedding_vectors, self.train_outputs[layer_name])
            
            # randomly select d dimension
            embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)

            # calculate multivariate Gaussian distribution
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            mean = torch.mean(embedding_vectors, dim=0).numpy()
            cov = torch.zeros(C, C, H * W).numpy()
            I = np.identity(C)
            for i in range(H * W):
                cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
            
            # save learned distribution
            learned_distribution = [mean, cov]
            save_feat_pickle(feat=learned_distribution, file_path=self.embedding_dir_path)
                

    def prediction(self):

        self.backbone.eval()

        self.pixel_gt_list.clear()
        self.img_gt_list.clear()

        #self.pixel_pred_list.clear()
        #self.img_pred_list.clear()

        if self.chosen_valid_loader.batch_size != 1:
            assert 'PaDim Evaluation, Batch Size should be Equal to 1'

        PaDim.dict_clear(self.test_outputs) 

        with torch.no_grad():
            for batch_id, batch in enumerate(self.chosen_valid_loader):
                img = batch['img'].to(self.device)
                mask = batch['mask'].to(self.device)
                label = batch['label'].to(self.device)
                self.img_gt_list.extend(label.cpu().detach().numpy())
                self.pixel_gt_list.extend(mask.cpu().detach().numpy())
                # Extract features from backbone
                self.features.clear()
                _ = self.backbone(img)

                #get the intermediate layer outputs
                for k,v in zip(self.test_outputs.keys(), self.features):
                    self.test_outputs[k].append(v.cpu().detach())
        
        for k, v in self.train_outputs.items():
            self.test_outputs[k] = torch.cat(v, 0)

        embedding_vectors = self.test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = PaDim.embedding_concate(embedding_vectors, self.train_outputs[layer_name])  
        
        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)

        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        for i in range(H * W):
            mean = self.train_outputs[0][:, i]
            conv_inv = np.linalg.inv(self.train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)
        
        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=img.size(2), mode='bilinear',
                                  align_corners=False).squeeze().numpy()
        
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)

        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        self.img_gt_list = np.asarray(self.img_gt_list)
        img_auroc = np_get_auroc(self.img_gt_list, img_scores) 

        # calculate pixel-level AUROC
        pixel_auroc = np_get_auroc(self.pixel_gt_list.flatten(), scores.flatten())

        return pixel_auroc, img_auroc


