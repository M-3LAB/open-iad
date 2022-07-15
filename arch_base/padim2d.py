import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from random import sample
import torch.nn.functional as F
import numpy as np

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

        self.feaaturs = []

        self.train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        self.test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])]) 

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
            self.train_outputs = [mean, cov]
                
        


    def prediction(self):
        pass