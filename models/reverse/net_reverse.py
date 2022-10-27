import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from models.reverse.blocks import *
from models.reverse.encoder import *  
from models.reverse.decoder import *
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['enc_wide_resnet_50_2', 'dec_wide_resnet_50_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def _resnet(
    arch: str,
    block: Type[Union[EncBasicBlock, EncBottleneck, DecBasicBlock, DecBottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    phase: str,
    **kwargs: Any
):
    if phase == 'encode':
        model = EncResNet(block, layers, **kwargs)
    elif phase == 'decode':
        model = DecResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        #for k,v in list(state_dict.items()):
        #    if 'layer4' in k or 'fc' in k:
        #        state_dict.pop(k)
        model.load_state_dict(state_dict)
    return model

def enc_wide_resnet_50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', EncBottleneck, [3, 4, 6, 3],
                   pretrained, progress, phase='encode', **kwargs)

def dec_wide_resnet_50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', DecBottleneck, [3, 4, 6, 3],
                   pretrained, progress, phase='decode', **kwargs)

def bn_layer(**kwargs):
    return BNLayer(AttnBottleneck, 3, **kwargs) 

class NetReverse(nn.Module):
    def __init__(self, args):
        super(NetReverse, self).__init__()
        self.args = args

        self.encoder = enc_wide_resnet_50_2(pretrained=True)
        self.decoder = dec_wide_resnet_50_2(pretrained=False)
        self.bn = bn_layer()
        

