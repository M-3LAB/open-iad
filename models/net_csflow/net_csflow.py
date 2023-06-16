import torch
import torch.nn.functional as F
from torch import nn
from efficientnet_pytorch import EfficientNet
from models.net_csflow.freia_funcs import *

class NetCSFlow(nn.Module):
    def __init__(self, args):
        super(NetCSFlow, self).__init__()
        self.args = args
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b5')
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.map_size = (self.args._image_size // 12, self.args._image_size // 12)
        self.kernel_sizes = [3] * (self.args._n_coupling_blocks - 1) + [5]
        self.density_estimator = self.get_cs_flow_model(input_dim=self.args._n_feat)

    def get_cs_flow_model(self, input_dim):
        nodes = list()
        nodes.append(InputNode(input_dim, self.map_size[0], self.map_size[1], name='input'))
        nodes.append(InputNode(input_dim, self.map_size[0] // 2, self.map_size[1] // 2, name='input2'))
        nodes.append(InputNode(input_dim, self.map_size[0] // 4, self.map_size[1] // 4, name='input3'))

        for k in range(self.args._n_coupling_blocks):
            if k == 0:
                node_to_permute = [nodes[-3].out0, nodes[-2].out0, nodes[-1].out0]
            else:
                node_to_permute = [nodes[-1].out0, nodes[-1].out1, nodes[-1].out2]

            nodes.append(Node(node_to_permute, ParallelPermute, {'seed': k}, name=F'permute_{k}'))
            nodes.append(Node([nodes[-1].out0, nodes[-1].out1, nodes[-1].out2], parallel_glow_coupling_layer,
                              {'clamp': self.args._clamp, 'F_class': CrossConvolutions,
                               'F_args': {'channels_hidden': self.args._fc_internal,
                                          'kernel_size': self.kernel_sizes[k], 'block_no': k}},
                              name=F'fc1_{k}'))

        nodes.append(OutputNode([nodes[-1].out0], name='output_end0'))
        nodes.append(OutputNode([nodes[-2].out1], name='output_end1'))
        nodes.append(OutputNode([nodes[-3].out2], name='output_end2'))
        nf = ReversibleGraphNet(nodes, n_jac=3)
        return nf

    def eff_ext(self, x, use_layer=36):
        x = self.feature_extractor._swish(self.feature_extractor._bn0(self.feature_extractor._conv_stem(x)))
        # Blocks
        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.feature_extractor._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == use_layer:
                return x

    def forward_features(self, x):
        y = list()
        for s in range(self.args._n_scales):
            feat_s = F.interpolate(x, size=(
            self.args._image_size // (2 ** s), self.args._image_size // (2 ** s))) if s > 0 else x
            feat_s = self.eff_ext(feat_s)
            y.append(feat_s)
        return y

    def forward_logits(self, y):
        z, log_jac_det = self.density_estimator(y), self.density_estimator.jacobian(run_forward=False)
        z = torch.cat([z[i].reshape(z[i].shape[0], -1) for i in range(len(z))], dim=1)
        log_jac_det = sum(log_jac_det)
        return z, log_jac_det

    def forward(self, x):
        # embeds = torch.cat([y[i].reshape(y[i].shape[0], -1) for i in range(len(y))], dim=1)
        # z, log_jac_det = self.density_estimator(y), self.density_estimator.jacobian(run_forward=False)
        # z = torch.cat([z[i].reshape(z[i].shape[0], -1) for i in range(len(z))], dim=1)
        # log_jac_det = sum(log_jac_det)
        y = self.forward_features(x)  # y(16, 512, 24/12/6, 24/12/6)
        z, log_jac_det = self.forward_logits(y)  # z(16, 229824)

        zz = self.density_estimator(y)
        yy = self.density_estimator(zz, rev=True)
        return y, z, log_jac_det


