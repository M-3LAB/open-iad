import torch
import torch.nn as nn

__all__ = ['GraphPropagation']

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