import torch
from torch import nn
import numpy as np
import argparse
import torch.nn.functional as F
from arch_base.base import ModelBase
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from loss_function.loss import SSIM, FocalLoss
import imgaug.augmenters as iaa
from data_io.augmentation.draem_aug import DraemAugData

__all__ = ['DNE', 'weights_init']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _DRAEM(nn.Module):
    def __init__(self, args, net, optimizer, scheduler):
        super(_DRAEM, self).__init__()
        self.args = args
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_l2 = nn.modules.loss.MSELoss()
        self.loss_ssim = SSIM()
        self.loss_focal = FocalLoss()

    def forward(self, epoch, inputs, labels, masks):
        rec_imgs, out_masks = self.net(inputs)
        out_masks_sm = torch.softmax(out_masks, dim=1)
        l2_loss = self.loss_l2(rec_imgs, inputs)
        ssim_loss = self.loss_ssim(rec_imgs, inputs)
        segment_loss = self.loss_focal(out_masks_sm, masks)
        loss = l2_loss + ssim_loss + segment_loss

        loss.backward()
        self.optimizer.step()

class DRAEM(ModelBase):
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
        self.scheduler = scheduler

        self.args = argparse.Namespace(**self.config)
        self.model = _DRAEM(self.args, self.net, optimizer, self.scheduler).to(self.device)
        self.dream_aug = DraemAugData(self.args.root_path + '/dtd/images', [self.args.data_size, self.args.data_size])

    def train_model(self, train_loaders, inf=''):
        self.net.train()

        for task_idx, train_loader in enumerate(train_loaders):
            print('run task: {}'.format(self.config['train_task_id'][task_idx]))

            for epoch in range(self.config['num_epochs']):
                for batch_id, batch in enumerate(train_loader):
                    inputs, masks, labels = self.dream_aug.transform_batch(batch['img'], batch['label'], batch['mask'])
                    self.model(epoch, inputs.to(self.device), labels.to(self.device), masks.to(self.device))

                self.scheduler.step()

    def prediction(self, valid_loader, task_id):
        self.net.eval()
        pixel_auroc, img_auroc = 0, 0

        seg_score_gt, dec_score_prediction = [], []
        with torch.no_grad():
            for batch_id, batch in enumerate(valid_loader):
                inputs = batch['img'].to(self.device)
                labels = batch['label'].numpy()

                seg_score_gt.append(labels)
                _, out_masks = self.net(inputs)
                out_masks_sm = torch.softmax(out_masks, dim=1)
                outs_mask_averaged = torch.nn.functional.avg_pool2d(out_masks_sm[:, 1:, :, :],
                                                                     21, stride=1, padding=21 // 2).cpu().detach().numpy()

                image_score = np.max(outs_mask_averaged)
                dec_score_prediction.append(image_score)

        dec_score_prediction = np.array(dec_score_prediction)
        seg_score_gt = np.array(seg_score_gt)
        img_auroc = roc_auc_score(seg_score_gt, dec_score_prediction)

        return pixel_auroc, img_auroc