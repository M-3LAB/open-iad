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