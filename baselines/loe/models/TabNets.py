# Latent Outlier Exposure for Anomaly Detection with Contaminated Data
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch.nn as nn

class TabTransformNet(nn.Module):
    def __init__(self, x_dim,h_dim,bias,num_layers):
        super(TabTransformNet, self).__init__()
        net = []
        input_dim = x_dim
        for _ in range(num_layers-1):
            net.append(nn.Linear(input_dim,h_dim,bias=bias))
            net.append(nn.ReLU())
            input_dim= h_dim
        net.append(nn.Linear(input_dim,x_dim,bias=bias))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = self.net(x)

        return out


class TabEncoder(nn.Module):
    def __init__(self, x_dim,h_dim,z_dim,bias,num_layers,batch_norm):

        super(TabEncoder, self).__init__()

        enc = []
        input_dim = x_dim
        for _ in range(num_layers - 1):
            enc.append(nn.Linear(input_dim, h_dim,bias=bias))
            if batch_norm:
                enc.append(nn.BatchNorm1d(h_dim,affine=bias))
            enc.append(nn.ReLU())
            input_dim = h_dim

        self.enc = nn.Sequential(*enc)
        self.fc = nn.Linear(input_dim, z_dim,bias=bias)
    def forward(self, x):

        z = self.enc(x)
        z = self.fc(z)

        return z

class TabNets():

    def _make_nets(self,x_dim,config):
        enc_nlayers = config['enc_nlayers']
        try:
            hdim = config['enc_hdim']
            zdim = config['latent_dim']
            trans_dim = config['trans_hdim']
        except:
            if 32<=x_dim <= 300:
                zdim = 32
                hdim = 64
                trans_dim = x_dim
            elif x_dim<32:
                zdim = 2 * x_dim
                hdim = 2 * x_dim
                trans_dim = x_dim
            else:
                zdim = 64
                hdim = 256
                trans_dim = x_dim
        trans_nlayers = config['trans_nlayers']
        num_trans = config['num_trans']
        batch_norm = config['batch_norm']

        enc = TabEncoder(x_dim, hdim,zdim, False,enc_nlayers,batch_norm)
        trans = nn.ModuleList(
            [TabTransformNet(x_dim, trans_dim, False, trans_nlayers) for _ in range(num_trans)])

        return enc,trans
