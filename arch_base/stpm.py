import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
from metrics.common.np_auc_precision_recall import *

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
        self.optimizer = torch.optim.SGD(self.backbone_student.parameters(), lr=self.config['lr'], 
                                         momentum=self.config['momentum'], 
                                         weight_decay=self.config['weight_decay'])
        
        self.pixel_gt_list = []
        self.img_gt_list = []
        self.pixel_pred_list = []
        self.img_pred_list = [] 

    def cal_anomaly_map(self, feat_teachers, feat_students, out_size=224):
        anomaly_map = np.ones([out_size, out_size])
        a_map_list = []
        for i in range(len(feat_teachers)):
            fs = feat_students[i]
            ft = feat_teachers[i]
            fs_norm = F.normalize(fs, p=2)
            ft_norm = F.normalize(ft, p=2)
            a_map = 1 - F.cosine_similarity(fs_norm, ft_norm)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear')
            a_map = a_map[0,0,:,:].to('cpu').detach().numpy()
            a_map_list.append(a_map)
            anomaly_map *= a_map
        return anomaly_map, a_map_list
        
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
    
    def cal_loss(self, feat_teachers, feat_students, criterion):
        total_loss = 0
        for i in range(len(feat_teachers)):
            fs = feat_students[i] 
            ft = feat_teachers[i]
            _, _, h, w = fs.shape
            fs_norm = F.normalize(fs, p=2) 
            ft_norm = F.normalize(ft, p=2)
            f_loss = (0.5/(w*h))*criterion(fs_norm, ft_norm)
            total_loss += f_loss
        
        return total_loss
    
    def train_epoch(self, inf=''):
        self.backbone_teacher.eval()
        self.backbone_student.train()

        self.get_layer_features()

        for task_idx, train_loader in enumerate(self.chosen_train_loaders):
            print('run task: {}'.format(self.config['chosen_train_task_ids'][task_idx]))
            for epoch in range(self.config['num_epochs']):
                for batch_id, batch in enumerate(train_loader):
                    img = batch['img'].to(self.device)    
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(True):
                        self.features_teacher.clear()
                        self.features_student.clear()

                        _  = self.backbone_teacher(img)
                        _ = self.backbone_student(img)

                        loss = self.cal_loss(feat_teachers=self.features_teacher,
                                             feat_students=self.features_student,
                                             criterion=self.criterion)
                        
                        loss.backward()

                        self.optimizer.step()
                
                infor = '\r{}[Epoch {} / {}]  [Loss: {:.4f}]'.format(
                           '', epoch+1, self.config['num_epochs'],  
                           float(loss.data))
                
                print(infor, flush=True, end='  ') 

    def prediction(self):

        self.backbone_teacher.eval()
        self.backbone_student.eval()

        self.pixel_gt_list.clear()
        self.img_gt_list.clear()
        self.pixel_pred_list.clear()
        self.img_pred_list.clear()

        for batch_id, batch in enumerate(self.chosen_valid_loader):
            img = batch['img'].to(self.device)
            mask = batch['mask']

            mask[mask>0.5] = 1
            mask[mask<=0.5] = 0
        
            with torch.set_grad_enabled(False):
                self.features_teacher.clear()
                self.features_student.clear()

                _ = self.features_teacher(img)
                _ = self.features_student(img)

                anomaly_map, _ = self.cal_anomaly_map(feat_teachers=self.features_teacher,
                                                      feat_students=self.features_students)                 

                self.pixel_pred_list.append(anomaly_map.ravel())
                self.pixel_gt_list.extend(mask.cpu().numpy().astype(int).ravel())
                self.img_gt_list.extend(np.max(mask.cpu().numpy().astype(int)))
                self.img_gt_list.extend(np.max(anomaly_map))
        
        pixel_auroc = np_get_auroc(self.pixel_gt_list, self.pixel_pred_list)
        img_auroc = np_get_auroc(self.img_gt_list, self.img_pred_list)

        return pixel_auroc, img_auroc




                        
