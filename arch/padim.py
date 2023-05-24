import torch
from collections import OrderedDict
from random import sample
import torch.nn.functional as F
import numpy as np
from arch.base import ModelBase
from tools.utils import *
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from torchvision import models

__all__ = ['PaDim']

class PaDim(ModelBase):
    def __init__(self, config):
        super(PaDim, self).__init__(config)
        self.config = config

        if self.config['net'] == 'resnet18': 
            self.net = models.resnet18(pretrained=True, progress=True).to(self.device)
            t_d, d = 448, 100
        if self.config['net'] == 'wide_resnet50':
            self.net = models.wide_resnet50_2(pretrained=True, progress=True).to(self.device)
            t_d, d = 1792, 550
            
        # random select d dimension 
        self.idx = torch.tensor(sample(range(0, t_d), d))
        self.features = []
        self.get_layer_features()
    
    def get_layer_features(self):
    
        def hook_t(module, input, output):
            self.features.append(output)
        
        self.net.layer1[-1].register_forward_hook(hook_t)
        self.net.layer2[-1].register_forward_hook(hook_t)
        self.net.layer3[-1].register_forward_hook(hook_t)
    
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

    def train_model(self, train_loader, task_id, inf=''):
        self.net.eval()
        self.train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        for _ in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                img = batch['img'].to(self.device) 

                with torch.no_grad():
                    _ = self.net(img)
                    
                # get the intermediate layer outputs
                for k, v in zip(self.train_outputs.keys(), self.features):
                    self.train_outputs[k].append(v.cpu().detach())
                # initialize hook outputs
                self.features = [] 

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
        self.train_outputs = [mean, cov]
        # save_feat_pickle(feat=self.train_outputs, file_path=self.embedding_dir_path + '/feature.npy')
                

    def prediction(self, valid_loader, task_id):
        self.net.eval()
        self.clear_all_list()
        self.test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])]) 

        for batch_id, batch in enumerate(valid_loader):
            
            img = batch['img'].to(self.device)
            mask = batch['mask'].numpy()
            label = batch['label'].numpy()       
               
            mask[mask >= 0.5] = 1
            mask[mask < 0.5] = 0
            self.img_gt_list.append(label)
            self.pixel_gt_list.append(mask[0, 0])
            self.img_path_list.append(batch['img_src'])
            # extract features from backbone
            with torch.no_grad():
                _ = self.net(img)

            # get the intermediate layer outputs
            for k, v in zip(self.test_outputs.keys(), self.features):
                self.test_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            self.features = [] 

        for k, v in self.test_outputs.items():
            self.test_outputs[k] = torch.cat(v, 0)

        embedding_vectors = self.test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = PaDim.embedding_concate(embedding_vectors, self.test_outputs[layer_name])  
        
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
        
        # normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)

        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        self.img_pred_list = img_scores
        self.pixel_pred_list = scores
