import torch
from models._patchcore.kcenter_greedy import KCenterGreedy 
import cv2
import torch.nn.functional as F
import numpy as np
from sklearn.random_projection import SparseRandomProjection
import faiss
import math
from scipy.ndimage import gaussian_filter
from arch_base.base import ModelBase
from tools.utils import *
import os

__all__ = ['GraphCore']

class GraphCore(ModelBase):

    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        super(GraphCore, self).__init__(config, device, file_path, net, optimizer, scheduler)

        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net

        self.model = self.net.model
        self.model.to(self.device)
        self.get_layer_features()

        self.features = []
        self.random_projector = SparseRandomProjection(n_components='auto', eps=0.9)
        self.embedding_coreset = np.array([])
        self.embedding_path = self.file_path + '/embed'
        create_folders(self.embedding_path)

        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def get_layer_features(self):

        def hook_t(module, input, output):
            self.features.append(output)

        self.model.backbone[9].register_forward_hook(hook_t)
        self.model.backbone[10].register_forward_hook(hook_t)
    
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
                #print(len(embeddings)) 
                embedding = GraphCore.embedding_concate(embeddings[0], embeddings[1])
                embedding_test = GraphCore.reshape_embedding(embedding.detach().numpy())
                embedding_test = np.array(embedding_test)

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

                # Reweighting i.e., equation(7) in paper
                max_min_distance = score_patches[:, 0]
                ind = np.argmax(max_min_distance)
                N_b = score_patches[ind]
                w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
                img_score = w * max(max_min_distance)

                # Because the feature map size from the layer 2 of wide-resnet 18 is 28
                #anomaly_map = max_min_distance.reshape((28, 28))
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
                self.img_pred_list.append(img_score)
                self.img_path_list.append(batch['img_src'])

                
                