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

__all__ = ['PatchCore2D']

class PatchCore2D():
    def __init__(self, config, train_loaders, valid_loaders, device, file_path):
        
        self.config = config
        self.train_loaders = train_loaders
        self.valid_loaders = valid_loaders
        self.device = device
        self.file_path = file_path

        self.chosen_train_loaders = [] 
        if self.config['chosen_train_task_ids'] is not None:
            for idx in range(len(self.config['chosen_train_task_ids'])):
                self.chosen_train_loaders.append(self.train_loaders[self.config['chosen_train_task_ids'][idx]])
        else:
            self.chosen_train_loaders = self.train_loaders
        
        self.chosen_valid_loader = self.valid_loaders[self.config['chosen_test_task_id']] 

        # Backbone model
        if config['backbone'] == 'resnet18':
            self.backbone = models.resnet18(pretrained=True, progress=True).to(self.device)
        elif config['backbone'] == 'wide_resnet50':
            self.backbone = models.wide_resnet50_2(pretrained=True, progress=True).to(self.device)
        else:
            raise NotImplementedError('This Pretrained Model Not Implemented Error')

        self.features = [] 
        self.get_layer_features(features=self.features)

        #TODO: Visualize Embeddings
        self.random_projector = SparseRandomProjection(n_components='auto', eps=0.9)

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

    @staticmethod 
    def reshape_embedding(embedding):
        embedding_list = []
        for k in range(embedding.shape[0]):
            for i in range(embedding.shape[2]):
                for j in range(embedding.shape[3]):
                    embedding_list.append(embedding[k, :, i, j])
                    #print(f'embedding[k,:,i,j].shape: {embedding[k,:,i,j].shape}')
        
        #print(f'embedding[k,:,i,j].shape: {embedding[4,:,2,3].shape}')
        return embedding_list
        
    def train_epoch(self, inf=''):
        
        self.backbone.eval()

        # When num_task is 15, per task means per class
        for task_idx, train_loader in enumerate(self.chosen_train_loaders):

            print('run task: {}'.format(task_idx))
            
            embedding_dir_path = os.path.join(self.file_path, 'embeddings', 
                                              str(self.config['chosen_train_task_ids'][task_idx]))

            create_folders(embedding_dir_path)
            #sampling_dir_path = os.path.join(self.file_path, 'samples', str(task_idx))
            #create_folders(sampling_dir_path)
            self.embeddings_list.clear()

            for _ in range(self.config['num_epoch']):
                for batch_id, batch in enumerate(train_loader):
                    print(f'batch id: {batch_id}')
                    #if self.config['debug'] and batch_id > self.config['batch_limit']:
                    #    break
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
                    #print(f'embedding.size: {embedding.size()}')
                    #print(f'embeddings[0].size: {embeddings[0].size()}')
                    #print(f'embeddings[1].size: {embeddings[1].size()}')

                    #print(f'embedding detach numpy.size: {embedding.detach().numpy().shape}')
                    embedding = PatchCore2D.reshape_embedding(embedding.detach().numpy())
                    #print(f'reshape embedding shape: {len(embedding)}')
                    self.embeddings_list.extend(embedding)

            # Sparse random projection from high-dimensional space into low-dimensional euclidean space
            total_embeddings = np.array(self.embeddings_list)
            self.random_projector.fit(total_embeddings)

            # Coreset subsampling
            # y refers to the label of total embeddings. X is good in training, so y=0
            selector = KCenterGreedy(X=total_embeddings, y=0)
            selected_idx = selector.select_batch(model=self.random_projector, 
                                                 already_selected=[],
                                                 N=int(total_embeddings.shape[0] * self.config['coreset_sampling_ratio']))

            self.embedding_coreset = total_embeddings[selected_idx]
        
            print('initial embedding size : ', total_embeddings.shape)
            print('final embedding size : ', self.embedding_coreset.shape)

            self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
            self.index.add(self.embedding_coreset) 
            faiss.write_index(self.index, os.path.join(embedding_dir_path, 'index.faiss'))
                    
    def prediction(self):

        self.backbone.eval()

        self.index = faiss.read_index(os.path.join(self.file_path, 'embeddings', 
                                      str(self.config['chosen_test_task_id']), 'index.faiss')) 
      

