import torch
import torch.nn as nn
from models.patchcore.kcenter_greedy import KCenterGreedy 
from torchvision import transforms
import cv2
from typing import List
from tools.utils import *
import os
import torch.nn.functional as F
import numpy as np
from sklearn.random_projection import SparseRandomProjection
import faiss
import math
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from tools.visualize import save_anomaly_map, vis_embeddings
from memory_augmentation.domain_generalization import feature_augmentation
from arch_base.base import ModelBase
from metrics.mvtec3d.au_pro import calculate_au_pro

__all__ = ['PatchCore']

class PatchCore(ModelBase):
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        super(PatchCore, self).__init__(config, device, file_path, net, optimizer, scheduler)
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net

        self.net.to(self.device)
        self.features = [] 
        self.get_layer_features()

        #TODO: Visualize Embeddings
        self.random_projector = SparseRandomProjection(n_components='auto', eps=0.9)

        self.pixel_gt_list = []
        self.img_gt_list = []
        self.pixel_pred_list = []
        self.img_pred_list = []
        self.img_path_list = []

        self.embeddings_list = []

        source_domain = ''
        if self.config['continual']:
            for i in self.config['train_task_id']:  
                source_domain = source_domain + str(self.config['train_task_id'][i])
        else:
            source_domain = str(self.config['train_task_id'][0])

        self.embedding_dir_path = os.path.join(self.file_path, 'embeddings', source_domain)
        create_folders(self.embedding_dir_path)
    
    def get_layer_features(self):

        def hook_t(module, input, output):
            self.features.append(output)
        
        #self.net.layer1[-1].register_forward_hook(hook_t)
        self.net.layer2[-1].register_forward_hook(hook_t)
        self.net.layer3[-1].register_forward_hook(hook_t)

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
        
    def train_model(self, train_loader, task_id, inf=''):
        self.net.eval()

        for _ in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                img = batch['img'].to(self.device)
                # Extract features from backbone
                self.features.clear()
                _ = self.net(img)

                embeddings = []
                for feat in self.features:
                    # Pooling for layer 2 and layer 3 features
                    pooling = torch.nn.AvgPool2d(3, 1, 1)
                    embeddings.append(pooling(feat))

                embedding = PatchCore.embedding_concate(embeddings[0], embeddings[1])
                embedding = PatchCore.reshape_embedding(embedding.detach().numpy())
                self.embeddings_list.extend(embedding)

                if self.config['fewshot']:
                    embeddings_rot = []
                    if self.config['fewshot_feat_aug']:
                        self.embed_rot = feature_augmentation(self.features, self.device)

                        for feat in self.embed_rot:
                            # Pooling for layer 2 and layer 3 features
                            pooling = torch.nn.AvgPool2d(3, 1, 1)
                            embeddings_rot.append(pooling(feat))

                        embedding_rot = PatchCore.embedding_concate(embeddings_rot[0], embeddings_rot[1])
                        embedding_rot = PatchCore.reshape_embedding(embedding_rot.detach().numpy())
                        self.embeddings_list.extend(embedding_rot)

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


    def prediction(self, valid_loader, task_id=None):
        self.net.eval()

        self.index = faiss.read_index(os.path.join(self.embedding_dir_path, 'index.faiss')) 
        
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, int(self.config['gpu_id']), self.index)
        
        self.pixel_gt_list.clear()
        self.img_gt_list.clear()
        self.pixel_pred_list.clear()
        self.img_pred_list.clear()
        self.img_path_list.clear()

        sampling_dir_path = os.path.join(self.file_path, 'samples', str(self.config['valid_task_id']))
        create_folders(sampling_dir_path)

        if valid_loader.batch_size != 1:
            assert 'PatchCore Evaluation, Batch Size should be Equal to 1'

        with torch.no_grad():
            for batch_id, batch in enumerate(valid_loader):
                img = batch['img'].to(self.device)
                mask = batch['mask'].to(self.device)
                label = batch['label'].to(self.device)
                # Extract features from backbone
                self.features.clear()
                _ = self.net(img)

                # Pooling for layer 2 and layer 3 features
                embeddings = []
                # print(f'Augmentat Features Size: {len(self.features)}')
                for feat in self.features:
                    # Pooling for layer 2 and layer 3 features
                    pooling = torch.nn.AvgPool2d(3, 1, 1)
                    embeddings.append(pooling(feat))

                embedding = PatchCore.embedding_concate(embeddings[0], embeddings[1])
                embedding_test = PatchCore.reshape_embedding(embedding.detach().numpy())
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

                mask[mask>=0.5] = 1
                mask[mask<0.5] = 0
                # print(mask.shape) 1 1 256 256
                mask_np = mask.cpu().numpy()[0,0].astype(int)
                self.pixel_gt_list.append(mask_np)
                self.pixel_pred_list.append(anomaly_map_cv)
                self.img_gt_list.append(label.cpu().numpy()[0])
                self.img_pred_list.append(img_score)
                self.img_path_list.append(batch['img_src'])

                #TODO: Anomaly Map Visualization
                if label == 0:
                    defect_type = 'norminal'
                else:
                    defect_type = 'anomaly'

                img_cv = PatchCore.torch_to_cv(img)
                save_anomaly_map(anomaly_map=anomaly_map_cv, input_img=img_cv,
                                 mask=mask_np*255, 
                                 file_path=os.path.join(sampling_dir_path, defect_type, str(batch_id)))
                
                
        # pixel_auroc = roc_auc_score(self.pixel_gt_list, self.pixel_pred_list) 
        # img_auroc = roc_auc_score(self.img_gt_list, self.img_pred_list)
        # pixel_ap = average_precision_score(self.pixel_gt_list, self.pixel_pred_list)
        # img_ap = average_precision_score(self.img_gt_list, self.img_pred_list)
        # au_pro, au_pro_curve = calculate_au_pro(self.pixel_gt_list_2d, self.pixel_pred_list_2d)
        # print('pixel_auroc: ', pixel_auroc, 'img_auroc: ', img_auroc, 'pixel_ap: ', pixel_ap, 'img_ap: ', img_ap, 'au_pro: ', au_pro)

        # return pixel_auroc, img_auroc



        
      

