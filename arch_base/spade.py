from metrics.common.np_auc_precision_recall import np_get_auroc
import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from tools.utilize import *
import os

__all__ = ['Spade']

class Spade():
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

        # Backbone model
        if config['backbone'] == 'resnet18':
            self.backbone = models.resnet18(pretrained=True, progress=True).to(self.device)
        elif config['backbone'] == 'wide_resnet50':
            self.backbone = models.wide_resnet50_2(pretrained=True, progress=True).to(self.device)
        else:
            raise NotImplementedError('This Pretrained Model Not Implemented Error')
        
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
        self.img_list = []
    
    @staticmethod
    def dict_clear(outputs):
        for key, value in outputs.items():
            if isinstance(value, list): 
                value.clear()
    
    def get_layer_features(self):
        
        def hook_t(module, input, output):
            self.features.append(output)
        
        self.backbone.layer1[-1].register_forward_hook(hook_t)
        self.backbone.layer2[-1].register_forward_hook(hook_t)
        self.backbone.layer3[-1].register_forward_hook(hook_t)
        self.backbone.avgpool.register_forward_hook(hook_t)
    
    @staticmethod
    def cal_distance_matrix(x, y):
        """Calculate Euclidean distance matrix with torch.tensor"""
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))
        return dist_matrix

    def train_epoch(self, inf=''):

        self.backbone.eval()
        # When num_task is 15, per task means per class
        self.get_layer_features(outputs=self.train_outputs)

        for task_idx, train_loader in enumerate(self.chosen_train_loaders):
            Spade.dict_clear(self.train_outputs)
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
                
                save_feat_pickle(feat=self.train_outputs, file_path=self.embedding_dir_path)
    
    def prediction(self):

        self.backbone.eval()
        self.pixel_gt_list.clear()
        self.img_gt_list.clear()
        self.img_list.clear()

        Spade.dict_clear(self.test_outputs)

        self.get_layer_features(outputs=self.test_outputs)

        for batch_id, batch in enumerate(self.chosen_valid_loader):
            img = batch['img'].to(self.device)
            mask = batch['mask'].to(self.device)
            label = batch['label'].to(self.device)

            self.img_list.extend(img.cpu().detach().numpy())
            self.img_gt_list.extend(label.cpu().detach().numpy())
            self.pixel_gt_list.extend(mask.cpu().detach().numpy())

            # Extract features from backbone
            with torch.no_grad():
                self.features.clear()
                _ = self.backbone(img)

            for k,v in zip(self.test_outputs.keys(), self.features):
                self.test_outputs[k].append(v)
        
        for k, v in zip(self.test_outputs.keys(), self.features):
            self.test_outputs[k] = torch.cat(v, 0)
        
        # Load train feature 
        self.train_outputs = load_feat_pickle(feat=self.train_outputs, 
                                              file_path=self.embedding_dir_path)

        # calculate distance matrix
        dist_matrix = Spade.cal_distance_matrix(torch.flatten(self.test_outputs['avgpool'], 1),
                                                torch.flatten(self.train_outputs['avgpool'], 1))
        
        # select K nearest neighbor and take advantage 
        topk_values, topk_indexes = torch.topk(dist_matrix, k=self.config['top_k'], 
                                               dim=1, largest=False)

        scores = torch.mean(topk_values, 1).cpu().detach().numpy()

        # calculate image-level AUROC
        img_auroc = np_get_auroc(self.img_gt_list, scores) 


        

