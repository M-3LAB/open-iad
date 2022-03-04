import torch
import torch.nn as nn

__all__ = ['hgnn']

class GNN_V1(nn.Module):
    def __init__(self, encode_dim=64, hidden_dim=64, n_prop_layer=5, g_repr_dim=128):
        super(GNN_V1, self).__init__()
 
        self.encoder = GraphEncoder(1, encode_dim // 2, encode_dim)
        self.prop_layers = nn.ModuleList([GraphPropLayer(encode_dim, hidden_dim) for i in range(n_prop_layer)])
        self.aggregator = GraphAggregator(encode_dim, hidden_dim, g_repr_dim)
        self.multiagg_layers = nn.ModuleList([GraphAggregator(encode_dim, hidden_dim, g_repr_dim) for i in range(n_prop_layer)])
        self.multihead_layer = MLP(g_repr_dim * 5, g_repr_dim * 3, g_repr_dim)
        # self.pridictor = GraphDecoder(n_node, encode_dim)

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