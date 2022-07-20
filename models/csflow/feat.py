import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

__all__ = ['FeatureExtractor', 'nf_forward']

class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()

        self.config = config

        if self.config['backbone'] == 'efficient_net':
            self.feature_extractor = EfficientNet.from_pretrained()
        else:
            raise NotImplementedError('This archichitecture has not been implemented yet')
    
    def eff_ext(self, x, use_layer=35):
        x = self.feature_extractor._swish(self.feature_extractor._bn0(self.feature_extractor._conv_stem(x)))
        # Blocks
        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.feature_extractor._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == use_layer:
                return x
    
    def forward(self, x):
        y = list()
        for s in range(self.config['n_scales']):
            feat_s = F.interpolate(x, size=(self.config['img_size'][0] // (2 ** s), 
                                            self.config['img_size'][1] // (2 ** s))) if s > 0 else x
            feat_s = self.eff_ext(feat_s)

            y.append(feat_s)
        return y

# Normalize Flow Forward
def nf_forward(model, inputs):
    return model(inputs), model.jacobian(run_forward=False)
