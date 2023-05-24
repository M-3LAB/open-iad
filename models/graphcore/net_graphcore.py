import torch
import torch.nn as nn
from models.graphcore.pyramid_vig import *
from models.graphcore.vig import *
from timm.models import create_model
from tools.utils import *
from torchprofile import profile_macs

__all__ = ['NetGraphCore']

def graphcore_ck_name(model_name, ck_path):
    model_name_splits = model_name.split('_')
    print(model_name_splits)
    if model_name_splits[0] == 'pvig':
        ck_name = ck_path+model_name_splits[0]+ '_' + model_name_splits[1] + '.pth.tar'
    elif model_name_splits[0] == 'vig':
        ck_name = ck_path+model_name_splits[0]+ '_' + model_name_splits[1] + '.pth'
    else:
        raise FileNotFoundError

    return ck_name

class NetGraphCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        create_folders(self.config.checkpoint_path)

        self.model = create_model(self.config.net,
                                  pretrained=self.config.pretrained,
                                  num_classes=1000,
                                  drop_rate=self.config.drop_rate,
                                  drop_path_rate=self.config.drop_path_rate,
                                  drop_block_rate=self.config.drop_block_rate,
                                  global_pool=self.config.gp,
                                  bn_tf=self.config.bn_tf,
                                  bn_momentum=self.config.bn_momentum,
                                  bn_eps=self.config.bn_eps)

        # Loading pretrained model  
        if self.config.checkpoint_path is not None:
            ck_name = graphcore_ck_name(self.config.net, self.config.checkpoint_path)
            print('Loading:', ck_name)
            state_dict = torch.load(ck_name)
            self.model.load_state_dict(state_dict, strict=False)
            print('Pretrain weights loaded')
        
        # Flops Calculation
        #print(self.model)
        if hasattr(self.model, 'default_cfg'):
            default_cfg = self.model.default_cfg
            input_size = [1] + list(default_cfg['input_size'])
        else:
            input_size = [1, 3, 224, 224]
        
        input = torch.randn(input_size)
        
        self.model.eval()
        macs = profile_macs(self.model, input)
        self.model.train()
        print('model flops:', macs, 'input_size:', input_size)
        
