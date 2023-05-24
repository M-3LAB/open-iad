import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import faiss
import math
import argparse
from models._patchcore.kcenter_greedy import KCenterGreedy 
from sklearn.random_projection import SparseRandomProjection
from scipy.ndimage import gaussian_filter
from arch.base import ModelBase
from models.graphcore.net_graphcore import NetGraphCore
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from tools.utils import *

__all__ = ['GraphCore']

class GraphCore(ModelBase):
    def __init__(self, config):
        super(GraphCore, self).__init__(config)
        self.config = config

        args = argparse.Namespace(**self.config)
        self.net = NetGraphCore(args)
        self.optimizer = create_optimizer(args, self.net.model)
        self.scheduler = create_scheduler(args, self.optimizer)
        self.model = self.net.model.to(self.device)
        self.features = []
        self.get_layer_features()

        self.random_projector = SparseRandomProjection(n_components='auto', eps=0.9)
        self.embedding_coreset = np.array([])
        self.embedding_path = self.config['file_path'] + '/embed'
        create_folders(self.embedding_path)

    def get_layer_features(self):

        def hook_t(module, input, output):
            self.features.append(output)

        self.model.backbone[self.config['layer_num_1']][-1].register_forward_hook(hook_t)
        self.model.backbone[self.config['layer_num_2']][-1].register_forward_hook(hook_t)

        #self.model.stem.register_forward_hook(hook_t)
        #self.model.stem.register_forward_hook(hook_t)
    
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
    
    @staticmethod 
    def reshape_embedding(embedding):
        embedding_list = []
        for k in range(embedding.shape[0]):
            for i in range(embedding.shape[2]):
                for j in range(embedding.shape[3]):
                    embedding_list.append(embedding[k, :, i, j])
        
        return embedding_list

    def train_model(self, train_loader, task_id, inf=''):
        self.model.eval()
        embeddings_list = []
        for _ in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                img = batch['img'].to(self.device)
                # Extract features from backbone
                self.features.clear()
                _ = self.model(img)

                embeddings = []
                for feat in self.features:
                    if self.config['local_smoothing']:
                        pooling = torch.nn.AvgPool2d(3, 1, 1)
                        embeddings.append(pooling(feat))
                    else:
                        embeddings.append(feat)

                embedding = GraphCore.embedding_concate(embeddings[0], embeddings[1])
                embedding = GraphCore.reshape_embedding(embedding.detach().numpy())
                embeddings_list.extend(embedding)
        
         # Sparse random projection from high-dimensional space into low-dimensional euclidean space
        total_embeddings = np.array(embeddings_list).astype(np.float32)
        self.random_projector.fit(total_embeddings)
        # Coreset subsampling
        # y refers to the label of total embeddings. X is good in training, so y=0
        selector = KCenterGreedy(X=total_embeddings, y=0)
        selected_idx = selector.select_batch(model=self.random_projector, 
                                             already_selected=[],
                                             N=int(total_embeddings.shape[0] * self.config['sampler_percentage']))
        if self.embedding_coreset.size == 0:
            self.embedding_coreset = total_embeddings[selected_idx]
        else:
            self.embedding_coreset = np.concatenate([self.embedding_coreset, total_embeddings[selected_idx]], axis=0)

        print('current task embedding size: ', total_embeddings.shape)
        print('coreset embedding size: ', self.embedding_coreset.shape)

        self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
        self.index.add(self.embedding_coreset)
        faiss.write_index(self.index, os.path.join(self.embedding_path, 'index.faiss'))

    def prediction(self, valid_loader, task_id):
        self.model.eval()
        self.clear_all_list()

        if valid_loader.batch_size != 1:
            assert 'GraphCore Evaluation, Batch Size should be equal to 1'
        
        with torch.no_grad():
            for batch_id, batch in enumerate(valid_loader):
                img = batch['img'].to(self.device)
                mask = batch['mask'].to(self.device)
                label = batch['label'].to(self.device)
                # Extract features from backbone
                self.features.clear()
                _ = self.model(img)

                embeddings = []
                
                for feat in self.features:
                    if self.config['local_smoothing']:
                        pooling = torch.nn.AvgPool2d(3, 1, 1)
                        embeddings.append(pooling(feat))
                    else:
                        embeddings.append(feat)
                    
                embedding = GraphCore.embedding_concate(embeddings[0], embeddings[1])
                embedding_test = GraphCore.reshape_embedding(embedding.detach().numpy())
                embedding_test = np.array(embedding_test)

                # Nearest Neighbour Search
                score_patches, _ = self.index.search(embedding_test, k=int(self.config['n_neighbours']))

                # No Reweighting, Directly Obtain max_min_distance
                max_min_distance = score_patches[:, 0]
                #print(f'max_min_distance: {max_min_distance}')
                img_score = max(max_min_distance)

                anomaly_map_size = math.sqrt(max_min_distance.shape[0])
                anomaly_map = max_min_distance.reshape(int(anomaly_map_size), int(anomaly_map_size))
                anomaly_map_resized = cv2.resize(anomaly_map, (self.config['data_crop_size'], self.config['data_crop_size']))
                anomaly_map_cv = gaussian_filter(anomaly_map_resized, sigma=4)

                mask[mask >= 0.5] = 1
                mask[mask < 0.5] = 0
                mask_np = mask.cpu().numpy()[0, 0].astype(int)
                self.pixel_gt_list.append(mask_np)
                self.pixel_pred_list.append(anomaly_map_cv)
                self.img_gt_list.append(label.cpu().numpy()[0])
                #print(img_score)
                self.img_pred_list.append(img_score)
                self.img_path_list.append(batch['img_src'])
            
            max_dis = np.max(np.array(self.pixel_pred_list))
            min_dis = np.min(np.array(self.pixel_pred_list))
            for i, pred_mask in enumerate(self.pixel_pred_list):
                self.pixel_pred_list[i] = (pred_mask - min_dis)/(max_dis - min_dis)


                
                