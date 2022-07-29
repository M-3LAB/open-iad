import torch
import torch.nn as nn
from models.cfa.efficientnet import EfficientNet as effnet
from models.cfa.resnet import wide_resnet50_2, resnet18
from models.cfa.vgg import vgg19_bn as vgg19

__all__ = ['CFA']

class CFA():

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
        
        if self.config['backbone'] == 'resnet18':
            self.backbone = resnet18()

    def train_on_epoch(self):
        pass

    def prediction(self):
        pass