import torch 
import torch.nn as nn
from models.graph.modules.mlp import MLP

__all__ = ['GraphDecoder']

class GraphDecoder(nn.Module):
    def __init__(self, n_node, encode_dim):
        super(GraphDecoder, self).__init__()
        self.n_node = n_node
        self.encode_dim = encode_dim
        self.mlp = MLP(encode_dim * 2, encode_dim, 2, layer_num=5)
        self.idx = self._get_idx().cuda()
        self.n = (self.n_node * self.n_node - self.n_node) // 2

    def _get_idx(self):
        _idx = [[[i, j] for i in range(j + 1, self.n_node)] for j in range(self.n_node)]
        _idx = [item for sublist in _idx for item in sublist]
        # idx_ = np.array(_idx)[:, [1, 0]].tolist()
        # idx = np.concatenate([_idx, idx_], axis=0)
        idx = _idx
        return torch.tensor(idx)

    def get_dmap(self, deg):
        idx = []
        for i in range(deg.size(0)):
            idx.append(self.idx + i * self.n_node)
        idx = torch.cat(idx, dim=0)

        deg = deg.view(-1, 1)
        y = deg[idx[:, 0]].view(-1, self.n).unsqueeze(2)
        z = deg[idx[:, 1]].view(-1, self.n).unsqueeze(2)

        return torch.cat([y, z], dim=-1)

    def forward(self, x):
        x = x.view(-1, self.n_node, self.encode_dim) # n*bs，20， 64
        idx = []
        for i in range(x.size(0)):
            idx.append(self.idx + i * self.n_node)
        idx = torch.cat(idx, dim=0)
        x = x.view(-1, self.encode_dim) # n*bs*20， 64
        y = x[idx[:, 0], :].view(-1, self.encode_dim).view(-1, self.n, self.encode_dim)
        z = x[idx[:, 1], :].view(-1, self.encode_dim).view(-1, self.n, self.encode_dim)

        x_cat = torch.cat([y, z], dim=-1)
        x = self.mlp(x_cat)

        return x