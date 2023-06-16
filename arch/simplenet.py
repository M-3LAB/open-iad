import numpy as np
from arch.base import ModelBase
from models.simplenet.simplenet import SimpleNet as _SimpleNet 
from torchvision import models

__all__ = ['SimpleNet']

class SimpleNet(ModelBase):
    def __init__(self, config):
        super(SimpleNet, self).__init__(config)
        self.config = config

        if self.config['net'] == 'wide_resnet50':
            self.net = models.wide_resnet50_2(pretrained=True, progress=True).to(self.device) 

        self.simplenet = _SimpleNet(self.device)
        self.simplenet.load(
            backbone=self.net,
            layers_to_extract_from=self.config['_layers_to_extract_from'],
            device=self.device,
            input_shape=(3, self.config['data_crop_size'], self.config['data_crop_size']),
            pretrain_embed_dimension=self.config['_pretrain_embed_dimension'],
            target_embed_dimension=self.config['_target_embed_dimension'],
            patchsize=self.config['_patchsize'],
            embedding_size=self.config['_embedding_size'],
            meta_epochs=self.config['num_epochs'],
            aed_meta_epochs=self.config['_aed_meta_epochs'],
            gan_epochs=self.config['_gan_epochs'],
            noise_std=self.config['_noise_std'],
            dsc_layers=self.config['_dsc_layers'],
            dsc_hidden=self.config['_dsc_hidden'],
            dsc_margin=self.config['_dsc_margin'],
            dsc_lr=self.config['_dsc_lr'],
            auto_noise=self.config['_auto_noise'],
            train_backbone=self.config['_train_backbone'],
            cos_lr=self.config['_cos_lr'],
            pre_proj=self.config['_pre_proj'],
            proj_layer_type=self.config['_proj_layer_type'],
            mix_noise=self.config['_mix_noise'],
        )
    
    def train_model(self, train_loader, task_id, inf=''):
        self.simplenet.train_discriminator(train_loader)

    def prediction(self, valid_loader, task_id):
        self.clear_all_list()

        scores, segmentations, labels_gt, masks_gt, img_srcs = self.simplenet.predict(valid_loader)

        scores = np.array(scores)
        min_scores = scores.min(axis=-1).reshape(-1, 1)
        max_scores = scores.max(axis=-1).reshape(-1, 1)
        scores = (scores - min_scores) / (max_scores - min_scores)
        scores = np.mean(scores, axis=0)

        segmentations = np.array(segmentations)
        min_scores = segmentations.reshape(len(segmentations), -1).min(axis=-1).reshape(-1, 1, 1, 1)
        max_scores = segmentations.reshape(len(segmentations), -1).max(axis=-1).reshape(-1, 1, 1, 1)
        segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        segmentations = np.mean(segmentations, axis=0)
        segmentations[segmentations >= 0.5] = 1
        segmentations[segmentations < 0.5] = 0
        segmentations = np.array(segmentations, dtype='uint8')
        masks_gt = np.array(masks_gt).squeeze().astype(int)

        self.pixel_gt_list = [mask for mask in masks_gt]
        self.pixel_pred_list = [seg for seg in segmentations]
        self.img_gt_list = labels_gt
        self.img_pred_list = scores
        self.img_path_list = img_srcs