import torch
import torch.nn as nn
from models.graphcore.pyramid_vig import *
from models.graphcore.vig import *
from timm.models import create_model
from tools.utils import *
import torchprofile

__all__ = ['NetGraphCore']

class NetGraphCore(nn.module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        create_folders(self.config.checkpoint_path)
        create_folders(self.config.pretrain_path)

        self.model = create_model(self.config.net,
                                  pretrained=self.config.pretrained,
                                  num_classes=1000,
                                  drop_rate=self.config.drop_rate,
                                  drop_path_rate=self.drop_path_rate,
                                  drop_block_rate=self.drop_block,
                                  global_pool=self.config.gp,
                                  bn_tf=self.bn_tf,
                                  bn_momentum=self.bn_momentum,
                                  bn_eps=self.bn_eps,
                                  checkpoint_path=self.checkpoint_path)

        # Loading pretrained model  
        if self.config.pretrain_path is not None:
            print('Loading:', self.config.pretrain_path)
            state_dict = torch.load(self.config.pretrain_path)
            self.model.load_state_dict(state_dict, strict=False)
            print('Pretrain weights loaded')
        
        # Flops Calculation
        print(self.model)
        if hasattr(self.model, 'default_cfg'):
            default_cfg = self.model.default_cfg
            input_size = [1] + list(default_cfg['input_size'])
        else:
            input_size = [1, 3, 224, 224]
        

    
    #def forward(self, x):
    #    pass
