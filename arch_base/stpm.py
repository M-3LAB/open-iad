import torch
import torch.nn as nn
from torchvision import models

__all__ = ['STPM']

class STPM():
    def __init__(self, config, train_loaders, valid_loaders, device,
                 file_path, train_fewshot_loaders=None):
        
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
        if self.config['backbone'] == 'resnet18':
            self.backbone_teacher = models.resnet(pretrained=True, progress=True).to(self.device) 
            self.backbone_students = models.resnet(pretrained=True, progress=True).to(self.device)

        elif self.config['backbone'] == 'wide_resnet50':
            self.backbone_teacher = models.wide_resnet50_2(pretrained=True, 
                                                           progress=True).to(self.device) 

            self.backbone_student = models.wide_resnet50_2(pretrained=True, 
                                                            progress=True).to(self.device)
        else:
            raise NotImplementedError('This Pretrained Model Not Implemented Error')

        self.features_teacher = []
        self.features_student = []

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.model_s.parameters(), lr=self.config['lr'], 
                                         momentum=self.config['momentum'], 
                                         weight_decay=self.config['weight_decay'])
        
    def get_layer_features(self):
    
        def hook_t(module, input, output):
            self.features_teacher.append(output)
        
        def hook_s(module, input, output):
            self.features_student.append(output)
    
        self.backbone_teacher.layer1[-1].register_forward_hook(hook_t)
        self.backbone_teacher.layer2[-1].register_forward_hook(hook_t)
        self.backbone_teacher.layer3[-1].register_forward_hook(hook_t)

        self.backbone_student.layer1[-1].register_forward_hook(hook_s)
        self.backbone_student.layer2[-1].register_forward_hook(hook_s)
        self.backbone_student.layer3[-1].register_forward_hook(hook_s)
    
    def train_epoch(self, inf=''):
        self.backbone_teacher.eval()
        self.backbone_student.train()

        self.get_layer_features()

        for task_idx, train_loader in enumerate(self.chosen_train_loaders):
            print('run task: {}'.format(self.config['chosen_train_task_ids'][task_idx]))
            for _ in range(self.config['num_epochs']):
                for batch_id, batch in enumerate(train_loader):
                    img = batch['img'].to(self.device)    
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(True):
                        self.features_teacher.clear()
                        self.features_student.clear()