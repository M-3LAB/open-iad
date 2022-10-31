import torch
import torch.nn.functional as F
import os
from tools.utils import create_folders
from arch_base.base import ModelBase
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
import copy

__all__ = ['STPM']

class STPM(ModelBase):
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        super(STPM, self).__init__(config, device, file_path, net, optimizer, scheduler)
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net_student = net.to(self.device) 
        self.net_teacher = copy.deepcopy(net).to(self.device) 

        if self.config['continual']:
            for i in self.config['train_task_id']:  
                source_domain = source_domain + str(self.config['train_task_id'][i])
        else:
            source_domain = str(self.config['train_task_id'][0])
        self.embedding_dir_path = os.path.join(self.file_path, 'embeddings', source_domain)
        create_folders(self.embedding_dir_path)

        self.features_teacher = []
        self.features_student = []
        self.get_layer_features()

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = optimizer

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
    
        self.net_teacher.layer1[-1].register_forward_hook(hook_t)
        self.net_teacher.layer2[-1].register_forward_hook(hook_t)
        self.net_teacher.layer3[-1].register_forward_hook(hook_t)

        self.net_student.layer1[-1].register_forward_hook(hook_s)
        self.net_student.layer2[-1].register_forward_hook(hook_s)
        self.net_student.layer3[-1].register_forward_hook(hook_s)
    
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
    
    def train_model(self, train_loader, task_id, inf=''):
        self.net_teacher.eval()
        self.net_student.train()

        for epoch in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                img = batch['img'].to(self.device)    
                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    self.features_teacher.clear()
                    self.features_student.clear()

                    _  = self.net_teacher(img)
                    _ = self.net_student(img)

                    loss = self.cal_loss(feat_teachers=self.features_teacher, feat_students=self.features_student,
                                            criterion=self.criterion)
                        
                    loss.backward()
                    self.optimizer.step()

    def prediction(self, valid_loader, task_id):
        self.net_teacher.eval()
        self.net_student.eval()
        self.clear_all_list()

        for batch_id, batch in enumerate(valid_loader):
            img = batch['img'].to(self.device)
            label = batch['label']
            mask = batch['mask']
            mask[mask>=0.5] = 1
            mask[mask<0.5] = 0

            with torch.set_grad_enabled(False):
                self.features_teacher.clear()
                self.features_student.clear()

                _ = self.net_teacher(img)
                _ = self.net_student(img)

                anomaly_map, _ = self.cal_anomaly_map(feat_teachers=self.features_teacher, feat_students=self.features_student,
                                                      out_size=self.config['data_crop_size'])                 

                self.pixel_pred_list.append(anomaly_map)
                self.pixel_gt_list.append(mask.cpu().numpy()[0,0].astype(int))
                self.img_pred_list.append(np.max(mask.cpu().numpy()).astype(int))
                self.img_gt_list.append(label.numpy()[0])
        
        # pixel_auroc = roc_auc_score(self.pixel_gt_list, self.pixel_pred_list)
        # img_auroc = roc_auc_score(self.img_gt_list, self.img_pred_list)

        # return pixel_auroc, img_auroc




                        
