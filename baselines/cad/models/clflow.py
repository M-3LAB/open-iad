import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34
from utils.freia_funcs import *

import torch
import torch.nn.functional as F
from torch import nn
from efficientnet_pytorch import EfficientNet
import numpy as np
from utils.freia_funcs import *

class NetCLFlow(nn.Module):
    def __init__(self, args, head_layers=[512, 512, 512, 512, 512, 512, 512, 512, 128]):
        super(NetCLFlow, self).__init__()
        self.args = args

        self.num_classes = args.train.num_classes
        self.backbone = resnet18(pretrained=self.args.model.pretrained)
        last_layer = 512
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1d(num_neurons))
            sequential_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons
        head = nn.Sequential(
            *sequential_layers
        )
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            head,
            nn.Linear(last_layer, self.num_classes)
        )

        self.feature_extractor = torch.nn.Sequential(*(list(self.backbone.children())[:8]))
        self.dim_redu = torch.nn.Sequential(*(list(self.backbone.children())[8:9]))

        self.map_size = (self.args.dataset.image_size // 12, self.args.dataset.image_size // 12)
        self.kernel_sizes = [3] * (self.args.model.n_coupling_blocks - 1) + [5]
        self.density_estimator = self.get_cs_flow_model(input_dim=self.args.model.n_feat)


    def get_cs_flow_model(self, input_dim):
        nodes = list()
        nodes.append(InputNode(input_dim, self.map_size[0], self.map_size[1], name='input'))
        nodes.append(InputNode(input_dim, self.map_size[0] // 2, self.map_size[1] // 2, name='input2'))
        nodes.append(InputNode(input_dim, self.map_size[0] // 4, self.map_size[1] // 4, name='input3'))

        for k in range(self.args.model.n_coupling_blocks):
            if k == 0:
                node_to_permute = [nodes[-3].out0, nodes[-2].out0, nodes[-1].out0]
            else:
                node_to_permute = [nodes[-1].out0, nodes[-1].out1, nodes[-1].out2]

            nodes.append(Node(node_to_permute, ParallelPermute, {'seed': k}, name=F'permute_{k}'))
            nodes.append(Node([nodes[-1].out0, nodes[-1].out1, nodes[-1].out2], parallel_glow_coupling_layer,
                              {'clamp': self.args.model.clamp, 'F_class': CrossConvolutions,
                               'F_args': {'channels_hidden': self.args.model.fc_internal,
                                          'kernel_size': self.kernel_sizes[k], 'block_no': k}},
                              name=F'fc1_{k}'))

        nodes.append(OutputNode([nodes[-1].out0], name='output_end0'))
        nodes.append(OutputNode([nodes[-2].out1], name='output_end1'))
        nodes.append(OutputNode([nodes[-3].out2], name='output_end2'))
        nf = ReversibleGraphNet(nodes, n_jac=3)
        return nf

    def forward_features(self, x):
        y = list()
        for s in range(self.args.model.n_scales):
            feat_s = F.interpolate(x, size=(
            self.args.dataset.image_size // (2 ** s), self.args.dataset.image_size // (2 ** s))) if s > 0 else x
            feat_s = self.feature_extractor(feat_s)
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

        embeds_dim2 = self.dim_redu(y[0])

        zz = self.density_estimator(y)
        yy = self.density_estimator(zz, rev=True)
        rev_embeds_dim2 = self.dim_redu(yy[0])

        return y, z, log_jac_det


class CLFlow(nn.Module):
    def __init__(self, args, net, optimizer, scheduler):
        super(CLFlow, self).__init__()
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net

    def forward(self, epoch, inputs, labels, one_epoch_embeds, task_wise_mean, task_wise_cov, t):
        self.optimizer.zero_grad()
        y = self.net.forward_features(inputs)
        z, log_jac_det = self.net.forward_logits(y)
        one_epoch_embeds.append(z.detach().cpu())


        y, z, log_jac_det = self.net(inputs)

        loss = torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - log_jac_det) / z.shape[1]

        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step(epoch)

    def training_epoch(self, density, one_epoch_embeds, task_wise_mean, task_wise_cov, task_wise_train_data_nums, t):
        pass
