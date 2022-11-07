import torch
import torch.nn as nn
import torch.nn.functional as F


        
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num=3, normalize=False, bias=True):
        super(MLP, self).__init__()
        self.normalize = normalize
        self.layer_num = layer_num
        self.conv_first = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.conv_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=bias) for i in range(layer_num - 2)])
        self.conv_out = nn.Linear(hidden_dim, output_dim, bias=bias)

        # self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.conv_first.weight, gain=gain)
        nn.init.xavier_normal_(self.conv_out.weight, gain=gain)

    def forward(self, x):
        x = self.conv_first(x)
        # x = F.dropout(x, 0.005)
        x = F.relu(x)

        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x)
            x = F.relu(x)

        x = self.conv_out(x)

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        return x

class GraphEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, outpu_size):
        super(GraphEncoder, self).__init__()
        self.node_encoder = MLP(input_size, hidden_size, outpu_size, layer_num=3, normalize=False)
        self.edge_encoder = MLP(input_size, hidden_size, outpu_size, layer_num=3, normalize=False)

    def forward(self, x):
        node, edge = x[0], x[1]
        node_feature = self.node_encoder(node)
        edge_feature = self.edge_encoder(edge)

        return node_feature, edge_feature

class GraphPropLayer(nn.Module):
    def __init__(self, node_state_dim, hidden_sizes):
        super(GraphPropLayer, self).__init__()
        self.message_net = MLP(node_state_dim * 3, hidden_sizes, node_state_dim, layer_num=3, normalize=False)
        self.reverse_message_net = MLP(node_state_dim * 3, hidden_sizes, node_state_dim, layer_num=3, normalize=False)
        self.gru_0 = nn.GRUCell(node_state_dim, node_state_dim)
        self.gru_1 = nn.GRUCell(node_state_dim, node_state_dim)
        self.gru_2 = nn.GRUCell(node_state_dim, node_state_dim)
        self.attn_fc = nn.Linear(3 * node_state_dim, 1, bias=False)


    def _graph_prop_once(self, node_states, from_idx, to_idx, message_net, edge_features, graph_idx):
        from_states = torch.index_select(node_states, 0, from_idx) # 126， 64  136  
        to_states = torch.index_select(node_states, 0, to_idx)

        edge_inputs = torch.cat([from_states, to_states, edge_features], dim=1)
        # edge_inputs = F.leaky_relu(self.attn_fc(edge_inputs)) * edge_inputs
        edge_states = message_net(edge_inputs).float()

        empty = torch.zeros(node_states.size()).type_as(node_states) 
        to_idx = to_idx.unsqueeze(1).expand_as(edge_states)
        edge_aggres = torch.scatter_add(empty, 0, to_idx, edge_states)

        return edge_aggres

    def _compute_aggregated_messages(self, x):
        node_states, from_idx, to_idx, edge_features, graph_idx = x
        aggregated_messages = self._graph_prop_once(node_states, from_idx, to_idx, self.message_net, edge_features, graph_idx)
        aggregated_messages += self._graph_prop_once(node_states, to_idx, from_idx, self.message_net, edge_features, graph_idx)

        return aggregated_messages 

    def _compute_node_update(self, inputs, h_pre):
        new_node_states1 = self.gru_0(inputs, h_pre)
        new_node_states2 = self.gru_1(h_pre, new_node_states1)
        new_node_states3 = self.gru_2(new_node_states1, new_node_states2)

        return new_node_states3

    def forward(self, x):
        aggregated_messages = self._compute_aggregated_messages(x)

        return self._compute_node_update(x[0], aggregated_messages)

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

class GNN(nn.Module):
    def __init__(self, encode_dim=64, hidden_dim=64, n_prop_layer=5, g_repr_dim=128):
        super(GNN, self).__init__()
 
        self.encoder = GraphEncoder(1, encode_dim // 2, encode_dim)
        self.prop_layers = nn.ModuleList([GraphPropLayer(encode_dim, hidden_dim) for i in range(n_prop_layer)])
        self.aggregator = GraphAggregator(encode_dim, hidden_dim, g_repr_dim)
        self.multiagg_layers = nn.ModuleList([GraphAggregator(encode_dim, hidden_dim, g_repr_dim) for i in range(n_prop_layer)])
        self.multihead_layer = MLP(g_repr_dim * 5, g_repr_dim * 3, g_repr_dim)

    def forward(self, x):
        node, edge, edge_index, graph_idx = x

        node_feature, edge_feature = self.encoder([node, edge])
        node_states = node_feature
        from_idx, to_idx = edge_index

        multiheads = []
        for i, layer in enumerate(self.prop_layers):
            node_states = layer([node_states, from_idx, to_idx, edge_feature, graph_idx]) # bs*4*n_node, hidden_dim
            multiheads.append(node_states)

        graph_repr = self.aggregator(node_states, graph_idx)

        return graph_repr
        
