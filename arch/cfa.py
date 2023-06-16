import torch
from arch.base import ModelBase
from models.cfa.net_cfa import NetCFA
from models.cfa.cfa import DSVDD
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

__all__ = ['CFA']

class CFA(ModelBase):
    def __init__(self, config):
        super(CFA, self).__init__(config)
        self.config = config
        self.net = NetCFA(self.config).resnet18.to(self.device)
    
    @staticmethod 
    def upsample(x, size, mode):
        return F.interpolate(x.unsqueeze(1), size=size, mode=mode, align_corners=False).squeeze().numpy()
    
    @staticmethod
    def gaussian_smooth(x, sigma=4):
        bs = x.shape[0]
        for i in range(0, bs):
            x[i] = gaussian_filter(x[i], sigma=sigma)

        return x
    
    @staticmethod
    def rescale(x):
        return (x - x.min()) / (x.max() - x.min())
    
    def train_model(self, train_loader, task_id, inf=''):
        self.net.eval()

        self.loss_fn = DSVDD(model=self.net, data_loader=train_loader,
                             cnn='resnet18', gamma_c=self.config['gamma_c'],
                             gamma_d=self.config['gamma_d'], device=self.device)
        self.loss_fn = self.loss_fn.to(self.device)
        self.loss_fn.train()

        params = [{'params' : self.loss_fn.parameters()},]
        optimizer = torch.optim.AdamW(params=params, lr=1e-3, weight_decay=5e-4,
                                      amsgrad=True)

        for epoch in range(self.config['num_epochs']): 
            for batch_id, batch in enumerate(train_loader):
                optimizer.zero_grad()
                img = batch['img'].to(self.device)
                p = self.net(img)

                loss, _ = self.loss_fn(p)
                loss.backward()
                optimizer.step()
            
    def prediction(self, valid_loader, task_id=None):
        self.loss_fn.eval()
        self.clear_all_list()
        heatmaps = None

        with torch.no_grad():
            for batch_id, batch in enumerate(valid_loader):
                img = batch['img'].to(self.device)
                label = batch['label']
                mask = batch['mask'].to(self.device)
                mask[mask >= 0.5] = 1
                mask[mask < 0.5] = 0

                self.img_gt_list.append(label.cpu().detach().numpy())
                self.pixel_gt_list.append(mask.cpu().detach().numpy()[0,0,:,:])
                self.img_path_list.append(batch['img_src'])

                p = self.net(img)

                _, score = self.loss_fn(p)
                heatmap = score.cpu().detach()
                heatmap = torch.mean(heatmap, dim=1) 
                heatmaps = torch.cat((heatmaps, heatmap), dim=0) if heatmaps != None else heatmap
       
        heatmaps = self.upsample(heatmaps, size=img.size(2), mode='bilinear') 
        heatmaps = self.gaussian_smooth(heatmaps, sigma=4)
        
        scores = self.rescale(heatmaps)
        for i in range(scores.shape[0]):
            self.pixel_pred_list.append(scores[i])
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        self.img_pred_list = img_scores