import torch
import torch.nn as nn
from models.graph.modules.mlp import MLP
from models.graph.modules.encoder import GraphEncoder
from models.graph.modules.decoder import GraphDecoder
from models.graph.modules.aggregator import GraphAggregator
from models.graph.modules.propagation import GraphPropagation
__all__ = ['HGNN']

class HGNN(nn.Module):
    def __init__(self, encode_dim=64, hidden_dim=64, n_prop_layer=5, g_repr_dim=128, n_node=None):
        super(HGNN, self).__init__()

        self.encode_dim = encode_dim
        self.hidden_dim = hidden_dim
        self.n_prop_layer = n_prop_layer
        self.g_repr_dim = g_repr_dim
        self.n_node = n_node
 
        self.encoder = GraphEncoder(1, encode_dim // 2, encode_dim)
        self.prop_layers = nn.ModuleList([GraphPropagation(encode_dim, hidden_dim) for _ in range(n_prop_layer)])
        self.aggregator = GraphAggregator(encode_dim, hidden_dim, g_repr_dim)
        self.multiagg_layers = nn.ModuleList([GraphAggregator(encode_dim, hidden_dim, g_repr_dim) for _ in range(n_prop_layer)])
        self.multihead_layer = MLP(g_repr_dim * 5, g_repr_dim * 3, g_repr_dim)
        self.predictor = GraphDecoder(n_node, encode_dim)

    def forward(self, x):
        node, edge, edge_index, graph_idx = x

        node_feature, edge_feature = self.encoder([node, edge])
        node_states = node_feature
        from_idx, to_idx = edge_index

        multiheads = []
        for i, layer in enumerate(self.prop_layers):
            node_states = layer([node_states, from_idx, to_idx, edge_feature, graph_idx]) # bs*4*n_node, hidden_dim
            # node_repr = self.multiagg_layers[i](node_states, graph_idx)
            multiheads.append(node_states)

        # node_states = torch.cat(multiheads, dim=-1)
        # node_states = self.multihead_layer(multiheads)
        # node_states = torch.mean(torch.stack(multiheads, dim=1), dim=1)
        graph_repr = self.aggregator(node_states, graph_idx)

        # dmap = self.pridictor(node_states)
        # dmap_label = self.pridictor.get_dmap(node_weight)

        # dist = torch.cosine_similarity(node_feature[:1], edge_feature[:1])

        return graph_repr