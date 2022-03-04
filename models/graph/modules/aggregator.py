import torch
import torch.nn as nn
from models.graph.modules.mlp import MLP

__all__ = ['GraphAggregator']

class GraphAggregator(nn.Module):
    def __init__(self, n_outdim, hidden_dim, n_repre):
        super(GraphAggregator, self).__init__()
        self.n_outdim = n_outdim
        self.n_repre = n_repre
        self.gated = nn.Sigmoid()
        self.mlp = MLP(n_outdim, hidden_dim, n_repre, layer_num=3)

    def forward(self, node_states, graph_idx):
        node_states = self.mlp(node_states)
        node_states = self.gated(node_states) * node_states

        n_graph = torch.max(graph_idx).long() + 1
        empty = torch.zeros(n_graph, node_states.size(1)).type_as(node_states)
        graph_idx = graph_idx.unsqueeze(1).expand_as(node_states)
        graph_repre = torch.scatter_add(empty, 0, graph_idx, node_states)

        return graph_repre