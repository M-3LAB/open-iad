import torch
from torch import nn
import numpy as np
import argparse
from arch.base import ModelBase
from models.dream.draem import NetDRAEM
from loss_function.focal import FocalLoss
from loss_function.ssim import SSIMLoss
from augmentation.draem_aug import DraemAugData
from optimizer.optimizer import get_optimizer

__all__ = ['DRAEM', 'weights_init']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class DRAEM(ModelBase):
    def __init__(self, config):
        super(DRAEM, self).__init__(config)
        self.config = config

        args = argparse.Namespace(**self.config)
        self.net = NetDRAEM(args).to(self.device)
        self.optimizer = get_optimizer(self.config, list(self.net.reconstructive_subnetwork.parameters()) + list(self.net.discriminative_subnetwork.parameters()))
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [args.num_epochs * 0.8, args.num_epochs * 0.9], gamma=args._gamma, last_epoch=-1)
        self.dream_aug = DraemAugData(self.config['root_path'] + '/dtd/images', [args.data_size, args.data_size])

        self.net.reconstructive_subnetwork.apply(weights_init)
        self.net.discriminative_subnetwork.apply(weights_init)

        self.loss_l2 = nn.modules.loss.MSELoss()
        self.loss_ssim = SSIMLoss()
        self.loss_focal = FocalLoss()

    def train_model(self, train_loader, task_id, inf=''):
        self.net.train()

        for epoch in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                inputs, masks, labels = self.dream_aug.transform_batch(batch['img'], batch['label'], batch['mask'])
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)

                rec_imgs, out_masks = self.net(inputs)

                out_masks_sm = torch.softmax(out_masks, dim=1)
                l2_loss = self.loss_l2(rec_imgs, inputs)
                ssim_loss = self.loss_ssim(rec_imgs, inputs)
                segment_loss = self.loss_focal(out_masks_sm, masks)
                loss = l2_loss + ssim_loss + segment_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            self.scheduler.step()

    def prediction(self, valid_loader, task_id):
        self.net.eval()
        self.clear_all_list()

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