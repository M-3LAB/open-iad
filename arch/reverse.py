import torch
import numpy as np
import argparse
from torch.nn import functional as F
from arch.base import ModelBase
from scipy.ndimage import gaussian_filter
from loss_function.reverse import reverse_loss
from models.reverse.net_reverse import NetReverse
from optimizer.optimizer import get_optimizer

__all__ = ['REVERSE']

class REVERSE(ModelBase):
    def __init__(self, config):
        super(REVERSE, self).__init__(config)
        self.config = config

        args = argparse.Namespace(**self.config) 
        self.net = NetReverse(args) 
        self.optimizer = get_optimizer(self.config, list(self.net.decoder.parameters()) + list(self.net.bn.parameters()))
        self.encoder = self.net.encoder.to(self.device)
        self.decoder = self.net.decoder.to(self.device)
        self.bn = self.net.bn.to(self.device)

    def train_model(self, train_loader, task_id, inf=''):
        self.encoder.eval() 
        self.bn.train()
        self.decoder.train()
        loss_list = []
        
        for epoch in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                img = batch['img'].to(self.device)

                inputs = self.encoder(img)
                outputs = self.decoder(self.bn(inputs))
                loss = reverse_loss(inputs, outputs)
                loss_list.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def prediction(self, valid_loader, task_id):
        self.encoder.eval()
        self.bn.eval()
        self.decoder.eval()
        self.clear_all_list()

        self.pixel_gt_list = []
        self.pixel_pred_list = []
        self.img_gt_list = []
        self.img_pred_list = []

        with torch.no_grad():
            for batch_id, batch in enumerate(valid_loader):
                img = batch['img'].to(self.device)
                mask = batch['mask']
                label = batch['label']
                inputs = self.encoder(img)
                outputs = self.decoder(self.bn(inputs))

                anomaly_map, _ = self.cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)

                mask[mask >= 0.5] = 1
                mask[mask < 0.5] = 0

                self.pixel_gt_list.append(mask.cpu().numpy()[0,0].astype(int))
                self.pixel_pred_list.append(anomaly_map)
                self.img_gt_list.append(label.numpy()[0])
                self.img_pred_list.append(np.max(anomaly_map))
                self.img_path_list.append(batch['img_src'])

    def cal_anomaly_map(self, fs_list, ft_list, out_size=256, amap_mode='full'):
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
