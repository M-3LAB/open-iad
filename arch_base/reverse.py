from models.reverse.loss import reverse_loss
import torch
import torch.nn as nn
from models.reverse.resnet import * 
from models.reverse.loss import *
import numpy as np
from torch.nn import functional as F
#from tools.visualize import *
from scipy.ndimage import gaussian_filter
from metrics.common.np_auc_precision_recall import *

__all__ = ['Reverse']

class Reverse():
    def __init__(self, config, train_loaders, valid_loaders,
                 device, file_path, train_fewshot_loaders=None):

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
        
        assert len(self.chosen_train_loaders) == 1, 'Only Support Single Transfer Single'

        self.chosen_valid_loader = self.valid_loaders[self.config['chosen_test_task_id']] 

        if self.config['fewshot']:
            assert self.train_fewshot_loaders is not None
            self.chosen_fewshot_loader = self.train_fewshot_loaders[self.config['chosen_test_task_id']]
        
        if self.config['chosen_test_task_id'] in self.config['chosen_train_task_ids']:
            assert self.config['fewshot'] is False, 'Changeover: test task id should not be the same as train task id'
        
        if self.config['backbone'] == 'wide_resnet50':
            self.encoder, self.bn = enc_wide_resnet_50_2(pretrained=True)
            self.decoder = dec_wide_resnet_50_2(pretrained=True)
        
        self.encoder = self.encoder.to(self.device)
        self.bn = self.bn.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.optimizer = torch.optim.Adam(list(self.decoder.parameters() + list(self.bn.parameters())),
                                          lr=self.config['lr'],
                                          betas=[self.config['beta1'], self.config['beta2']])
        
        self.pixel_gt_list = []
        self.img_gt_list = []
        self.pixel_pred_list = []
        self.img_pred_list = []
        self.aupro_list = []

    def train_epoch(self):
        #Encoder Do Not Need to Train
        self.encoder.eval()

        #BN and Decoder
        self.bn.train()
        self.decoder.train()
        loss_list = []
        # When num_task is 15, per task means per class
        for task_idx, train_loader in enumerate(self.chosen_train_loaders):
            print('run task: {}'.format(self.config['chosen_train_task_ids'][task_idx]))

            for epoch in range(self.config['num_epochs']):
                for batch_id, batch in enumerate(train_loader):
                    print(f'batch id: {batch_id}')
                    img = batch['img'].to(self.device)
                    #mask = batch['mask'].to(self.device)

                    inputs = self.encoder(img)
                    outputs = self.decoder(self.bn(inputs))
                    loss = reverse_loss(inputs, outputs)
                    loss_list.append(loss)


                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                infor = '\r{}[Epoch {} / {}] [Batch {}/{}] [Loss: {:.4f}]'.format(
                            '', epoch+1, self.config['num_epochs'], batch_id+1, batch, 
                            np.mean(loss.item()))

                print(infor, flush=True, end='  ') 

    def prediction(self):

        self.encoder.eval()
        self.bn.eval()
        self.decoder.eval()

        self.pixel_gt_list.clear()
        self.img_gt_list.clear()
        self.pixel_pred_list.clear()
        self.img_pred_list.clear()
        self.aupro_list.clear()

        with torch.no_grad():
            for batch_id, batch in enumerate(self.chosen_valid_loader):
                img = batch['img'].to(self.device)
                mask = batch['mask'].to(self.device)
                label = batch['label'].to(self.device)

                anomaly_map, _ = self.cal_anomaly_map()
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)

                mask[mask>0.5] = 1
                mask[mask<=0.5] = 0

                if label.item()!= 0:
                    pass
    
    def cal_anomaly_map(self, fs_list, ft_list, out_size=224, amap_mode='full'):
        if amap_mode == 'mul':
            anomaly_map = np.ones([out_size, out_size])
        else:
            anomaly_map = np.zeros([out_size, out_size])

        a_map_list = []
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            #fs_norm = F.normalize(fs, p=2)
            #ft_norm = F.normalize(ft, p=2)
            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
            a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
            a_map_list.append(a_map)
            if amap_mode == 'mul':
                anomaly_map *= a_map
            else:
                anomaly_map += a_map
        return anomaly_map, a_map_list