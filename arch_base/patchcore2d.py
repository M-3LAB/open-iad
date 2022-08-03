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
import kornia.geometry.transform as kt

__all__ = ['PatchCore2D']

class PatchCore2D():
    def __init__(self, config, train_loaders, valid_loaders, device, file_path, train_fewshot_loaders=None):
        
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

        if self.config['fewshot'] or self.config['fewshot_normal']:
        #     assert self.train_fewshot_loaders is not None
            self.chosen_fewshot_loader = self.train_fewshot_loaders[self.config['chosen_test_task_id']]
        
        # if self.config['chosen_test_task_id'] in self.config['chosen_train_task_ids']:
        #     assert self.config['fewshot'] is False, 'Changeover: test task id should not be the same as train task id'

        # Backbone model
        if config['backbone'] == 'resnet18':
            self.backbone = models.resnet18(pretrained=True, progress=True).to(self.device)
        elif config['backbone'] == 'wide_resnet50':
            self.backbone = models.wide_resnet50_2(pretrained=True, progress=True).to(self.device)
        else:
            raise NotImplementedError('This Pretrained Model Not Implemented Error')

        self.features = [] 
        self.get_layer_features()

        #TODO: Visualize Embeddings
        self.random_projector = SparseRandomProjection(n_components='auto', eps=0.9)

        self.pixel_gt_list = []
        self.img_gt_list = []
        self.pixel_pred_list = []
        self.img_pred_list = []

        self.embeddings_list = []

        for i in range(len(self.config['chosen_train_task_ids'])):
            if i == 0:
                source_domain = str(self.config['chosen_train_task_ids'][0])
            else:
                source_domain = source_domain + str(self.config['chosen_train_task_ids'][i])

        #target_domain = str(self.config['chosen_test_task_id'])
        self.embedding_dir_path = os.path.join(self.file_path, 'embeddings', 
                                          source_domain)
        create_folders(self.embedding_dir_path)
    
    def get_layer_features(self):

        def hook_t(module, input, output):
            self.features.append(output)
        
        #self.backbone.layer1[-1].register_forward_hook(hook_t)
        self.backbone.layer2[-1].register_forward_hook(hook_t)
        self.backbone.layer3[-1].register_forward_hook(hook_t)

    def feature_augmentation(self):
        assert len(self.features) > 0, 'Feature Augmentation should be done in Original Features'
        #print(len(self.features))
        #angles_list = [45, 90, 135, 180, 225, 270, 315, 360]
        angle = torch.tensor([90.]).to(self.device)
        rot_feat_1 = kt.rotate(self.features[0], angle) 
        rot_feat_2 = kt.rotate(self.features[1], angle) 
        #print(rot_feat.size())
        self.features.append(rot_feat_1)
        self.features.append(rot_feat_2)
        #for feat in self.features:
        #    angle = torch.tensor([90.]).to(self.device)
        #    rot_feat = kt.rotate(feat, angle) 
        #    #print(rot_feat.size())
        #    self.features.append(rot_feat)

            #for angle in angles_list:
            #    feat = feat.cpu()
            #    angle = torch.tensor([float(angle)])
            #    #print(angle)
            #    rot_feat = kt.rotate(feat, angle)
            #    #print(rot_feat.size())
            #    out.append(rot_feat)


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
        
        return embedding_list
        
    def train_epoch(self, inf=''):
        
        self.backbone.eval()

        if self.config['fewshot_normal']:
            pass
        else:
            # When num_task is 15, per task means per class
            for task_idx, train_loader in enumerate(self.chosen_train_loaders):
                print('run task: {}'.format(self.config['chosen_train_task_ids'][task_idx]))
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

                        embeddings = []

                        for feat in self.features:
                            # Pooling for layer 2 and layer 3 features
                            pooling = torch.nn.AvgPool2d(3, 1, 1)
                            embeddings.append(pooling(feat))

                        embedding = PatchCore2D.embedding_concate(embeddings[0], embeddings[1])

                        embedding = PatchCore2D.reshape_embedding(embedding.detach().numpy())
                        self.embeddings_list.extend(embedding)

        if self.config['fewshot'] or self.config['fewshot_normal']:
            print('Fewshot Processing')
            #print(f'The length of fewshot loader: {len(self.chosen_fewshot_loader)}')
            for _ in range(self.config['num_epochs']):
                for batch_id, batch in enumerate(self.chosen_fewshot_loader):
                    # print(f'fewshot batch id: {batch_id}')
                    #if self.config['debug'] and batch_id > self.config['batch_limit']:
                    #    break
                    img = batch['img'].to(self.device)
                    #mask = batch['mask'].to(self.device)

                    # Extract features from backbone
                    self.features.clear()
                    _ = self.backbone(img)

                    print(self.features[1].size())
                    # Pooling for layer 2 and layer 3 features
                    embeddings = []
                    if self.config['feat_aug']:
                        print('test')
                        self.feature_augmentation()

                    print(f'Augmentat Features Size: {len(self.features)}')

                    for feat in self.features:
                        # Pooling for layer 2 and layer 3 features
                        pooling = torch.nn.AvgPool2d(3, 1, 1)
                        embeddings.append(pooling(feat))

                    embedding = PatchCore2D.embedding_concate(embeddings[0], embeddings[1])

                    embedding = PatchCore2D.reshape_embedding(embedding.detach().numpy())
                    self.embeddings_list.extend(embedding)
            
        # Sparse random projection from high-dimensional space into low-dimensional euclidean space
        total_embeddings = np.array(self.embeddings_list).astype(np.float32)
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
        faiss.write_index(self.index, os.path.join(self.embedding_dir_path, 'index.faiss'))
                    
        # visualize embeddings
        if self.config['vis_em']:
            print('visualize embeddings')
            total_labels = np.array([i // 784 for i in range(total_embeddings.shape[0])])
            print(total_labels)
            embedding_label = total_labels[selected_idx]
            embedding_data = self.embedding_coreset
            print(embedding_label)
            vis_embeddings(embedding_data, embedding_label, self.config['fewshot_exm'], '{}/vis_embedding.png'.format(self.file_path))


    def prediction(self):

        self.backbone.eval()

        self.index = faiss.read_index(os.path.join(self.embedding_dir_path, 'index.faiss')) 
        
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, int(self.config['gpu_id']), self.index)
        
        self.pixel_gt_list.clear()
        self.img_gt_list.clear()
        self.pixel_pred_list.clear()
        self.img_pred_list.clear()

        sampling_dir_path = os.path.join(self.file_path, 'samples', str(self.config['chosen_test_task_id']))
        create_folders(sampling_dir_path)

        if self.chosen_valid_loader.batch_size != 1:
            assert 'PatchCore Evaluation, Batch Size should be Equal to 1'

        with torch.no_grad():
            for batch_id, batch in enumerate(self.chosen_valid_loader):
                img = batch['img'].to(self.device)
                mask = batch['mask'].to(self.device)
                label = batch['label'].to(self.device)
                # Extract features from backbone
                self.features.clear()
                _ = self.backbone(img)

                # Pooling for layer 2 and layer 3 features
                embeddings = []
                for feat in self.features:
                    pooling = torch.nn.AvgPool2d(3, 1, 1)
                    embeddings.append(pooling(feat))

                embedding_test = PatchCore2D.embedding_concate(embeddings[0], embeddings[1])
                embedding_test = PatchCore2D.reshape_embedding(embedding_test.detach().numpy())
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

                mask_np = mask.cpu().numpy()[0,0].astype(int)
                self.pixel_gt_list.extend(mask_np.ravel())
                self.pixel_pred_list.extend(anomaly_map_cv.ravel())
                self.img_gt_list.append(label.cpu().numpy()[0])
                self.img_pred_list.append(img_score)

                #TODO: Anomaly Map Visualization
                if label == 0:
                    defect_type = 'norminal'
                else:
                    defect_type = 'anomaly'

                img_cv = PatchCore2D.torch_to_cv(img)
                save_anomaly_map(anomaly_map=anomaly_map_cv, input_img=img_cv,
                                 mask=mask_np*255, 
                                 file_path=os.path.join(sampling_dir_path, defect_type, str(batch_id)))
                
                
        pixel_auroc = np_get_auroc(self.pixel_gt_list, self.pixel_pred_list) 
        img_auroc = np_get_auroc(self.img_gt_list, self.img_pred_list)

        return pixel_auroc, img_auroc



        
      

