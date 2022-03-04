import torch
import torch.nn as nn
from models.graph.modules.mlp import MLP

__all__ = ['GraphEncoder']

class GraphEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GraphEncoder, self).__init__()
        self.node_encoder = MLP(input_size, hidden_size, output_size, layer_num=3, normalize=False)
        self.edge_encoder = MLP(input_size, hidden_size, output_size, layer_num=3, normalize=False)

    def forward(self, x):
        node, edge = x[0], x[1]
        node_feature = self.node_encoder(node)
        edge_feature = self.edge_encoder(edge)

        return node_feature, edge_feature