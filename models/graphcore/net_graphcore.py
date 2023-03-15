import torch.nn as nn
from models.graphcore.pyramid_vig import *
from models.graphcore.vig import *
from timm.models import create_model
from tools.utils import *

__all__ = ['NetGraphCore']

class NetGraphCore(nn.module):
    def __init__(self, config):
        super().__init__()
        self.config = config
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
        
        if self.config.pretrain_path is not None:
            pass

        

    
    #def forward(self, x):
    #    pass
