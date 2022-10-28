import torch
import torch.n as nn
from arch_base.base import ModelBase
from torchvision import models
from models.favae.func import EarlyStop,AverageMeter, feature_extractor, print_log
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

__all__ = ['FAVAE']

class FAVAE(ModelBase):
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.teacher = models.vgg16(pretrained=True).to(self.device)
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.early_stop = EarlyStop(patience=20, save_name='favae.pt')
        self.mse_criterion = nn.MSELoss(reduction='sum')
    
    def train_model(self, train_loader, inf=''):
        self.net.train()
        self.teacher.eval()
        losses = AverageMeter()
        for epoch in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                img = batch['img'].to(self.device)
                z, output, mu, log_var = self.model(img)
                # get model's intermediate outputs
                s_activations, _ = feature_extractor(z, self.net.decode, target_layers=['10', '16', '22'])
                t_activations, _ = feature_extractor(z, self.teacher.features, target_layers=['7', '14', '21'])

                self.optimizer.zero_grad()
                mse_loss = self.mse_criterion(output, img)
                kld_loss = 0.5 * torch.sum(-1 - log_var + torch.exp(log_var) + mu**2)
                for i in range(len(s_activations)):
                    s_act = self.net.adapter[i](s_activations[-(i + 1)])
                    mse_loss += self.mse_criterion(s_act, t_activations[i])
                loss = mse_loss + self.config['kld_weight'] * kld_loss
                losses.update(loss.sum().item(), img.size(0))

                loss.backward()
                self.optimizer.step()
            
            infor = '\r{}[Epoch {}/{}] Loss: {:.6f}'.format(inf, epoch, self.config['num_epochs'],
                                                            losses.avg) 
            print(infor, flush=True, end='')

    def prediction(self, valid_loader, task_id=None):
        self.net().eval()
        self.teacher.eval()
        scores = []
        test_imgs = []
        gt_list = []
        gt_mask_list = []
        recon_imgs = []

        with torch.no_grad():
            for batch_id, batch in enumerate(valid_loader):
                img = batch['img'].to(self.device)
                z, output, mu, log_var = self.model(img)
                s_activations, _ = feature_extractor(z, self.model.decode, target_layers=['10', '16', '22']) 
                t_activations, _ = feature_extractor(img, self.teacher.features, target_layers=['7', '14', '21'])

                score = self.mse_criterion(output, img).sum(1, keepdim=True)

                for i in range(len(s_activations)):
                    s_act = self.net.adapter[i](s_activations[-(i + 1)])
                    mse_loss = self.mse_criterion(s_act, t_activations[i]).sum(1, keepdim=True)
                    score += F.interpolate(mse_loss, size=img.size(2), mode='bilinear', align_corners=False)
                
                score = score.squeeze().cpu().numpy()

                for i in range(score.shape[0]):
                    score[i] = gaussian_filter(score[i], sigma=4)

                scores.extend(score)
                recon_imgs.extend(output.cpu().numpy())

        
        scores = np.asarray(scores)

        max_anomaly_score = scores.max()
        min_anomaly_score = scores.min()

        scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

        gt_mask = np.asarray(gt_mask_list)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten()) 
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        pixel_auroc = roc_auc_score(gt_mask.flatten(), scores.flatten())

        return pixel_auroc, _