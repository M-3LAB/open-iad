import torch
from torch import nn
import numpy as np
import argparse
from arch_base.base import ModelBase
from loss_function.focal_loss import FocalLoss
from loss_function.ssim_loss import SSIMLoss
from data_io.augmentation.draem_aug import DraemAugData


__all__ = ['DRAEM', 'weights_init']

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
        self.loss_ssim = SSIMLoss()
        self.loss_focal = FocalLoss()

        self.net.apply(weights_init)
        
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
        super(DRAEM, self).__init__(config, device, file_path, net, optimizer, scheduler)
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
        self.scheduler = scheduler

        self.args = argparse.Namespace(**self.config)
        self.model = _DRAEM(self.args, self.net, optimizer, self.scheduler).to(self.device)
        self.dream_aug = DraemAugData(self.args.root_path + '/dtd/images', [self.args.data_size, self.args.data_size])

    def train_model(self, train_loader, task_id, inf=''):
        self.net.train()

        for epoch in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                inputs, masks, labels = self.dream_aug.transform_batch(batch['img'], batch['label'], batch['mask'])
                self.model(epoch, inputs.to(self.device), labels.to(self.device), masks.to(self.device))

            self.scheduler.step()

    def prediction(self, valid_loader, task_id):
        self.net.eval()
        self.cal_metric_all()

        with torch.no_grad():
            for batch_id, batch in enumerate(valid_loader):
                inputs = batch['img'].to(self.device)
                labels = batch['label'].numpy()
                mask = batch['mask'].numpy()

                _, out_masks = self.net(inputs)
                out_masks_sm = torch.softmax(out_masks, dim=1)
                out_mask_cv = out_masks_sm[0, 1, :, :].detach().cpu().numpy()
                outs_mask_averaged = torch.nn.functional.avg_pool2d(out_masks_sm[:, 1:, :, :],
                                                                     21, stride=1, padding=21 // 2).cpu().detach().numpy()
                image_score = np.max(outs_mask_averaged)
                self.pixel_pred_list.append(out_mask_cv)
                self.img_pred_list.append(image_score)

                mask[mask >= 0.5] = 1
                mask[mask < 0.5] = 0
                mask_np = mask[0, 0].astype(int)
                self.pixel_gt_list.append(mask_np)
                self.img_gt_list.append(labels[0])
                self.img_path_list.append(batch['img_src'])