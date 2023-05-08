import torch
import torch.nn as nn
from arch_base.base import ModelBase
from models.simplenet import simplenet

__all__ = ['SimpleNet']

class SimpleNet(ModelBase):

    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        super(SimpleNet, self).__init__(config, device)

        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
    
        self.simplenet = simplenet.SimpleNet(self.device)
        self.simplenet.load(
            backbone=self.net,
            layers_to_extract_from=self.config['layers_to_extract_from'],
            device=self.device,
            input_shape=self.config['data_size'],
            pretrain_embed_dimension=self.config['_pretrain_embed_dimension'],
            target_embed_dimension=self.config['_target_embed_dimension'],
            patchsize=self.config['_patchsize'],
            embedding_size=self.config['_embedding_size'],
            meta_epochs=self.config['_meta_epochs'],
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
        return simplenet
    
    def train_model(self, train_loader, task_id, inf=''):
     
        pass

    def prediction(self, valid_loader, task_id):
        pass