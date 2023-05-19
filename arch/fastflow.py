import torch
from arch.base import ModelBase
from models.fastflow.func import AverageMeter
import numpy as np


__all__ = ['FastFlow']

class FastFlow(ModelBase):
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        super(FastFlow, self).__init__(config, device, file_path, net, optimizer, scheduler)
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
        self.net.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_model(self, train_loader, inf=''):
        self.net.train()
        self.clear_all_list()
        loss_meter = AverageMeter() 
        for epoch in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                img = batch['img'].to(self.device)
                ret = self.net(img)
                loss = ret['loss']
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_meter.update(loss.item())
    
    def prediction(self, valid_loader, task_id=None):
        self.net.eval()
        self.clear_all_list()
        
        with torch.no_grad():
            for batch_id, batch in enumerate(valid_loader):
                img = batch['img'].to(self.device)
                mask = batch['mask']
                label = batch['label']
                mask[mask>=0.5] = 1
                mask[mask<0.5] = 0
                mask_np = mask.numpy()[0,0].astype(int)

                ret = self.net(img)

                outputs = ret["anomaly_map"].cpu().detach().numpy()
                self.pixel_gt_list.append(mask_np)
                self.pixel_pred_list.append(outputs[0,0,:,:])
                self.img_gt_list.append(label.numpy()[0])
                self.img_pred_list.append(np.max(outputs))
                self.img_path_list.append(batch['img_src'])
        