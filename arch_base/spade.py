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
        
        self.feaaturs = []

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
