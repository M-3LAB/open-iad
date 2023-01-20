import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNNet(torch.nn.Module):
    def __init__(self,input_features):
        super(GCNNet,self).__init__()
        self.conv1 = GCNConv(input_features, 256)
        self.conv2 = GCNConv(256,512)
        self.conv3 = GCNConv(512, 256)
        self.conv4 = GCNConv(256,64)
        self.conv5 = GCNConv(64,2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv5(x, edge_index)

        return F.log_softmax(x, dim=1)

class GCNNet1(torch.nn.Module):
    def __init__(self,input_features):
        super(GCNNet1,self).__init__()
        self.conv1 = GCNConv(input_features, 64)
        self.conv2 = GCNConv(64,128)
        self.conv3 = GCNConv(128,64)
        self.conv4 = GCNConv(64,16)
        self.conv5 = GCNConv(16,2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv5(x, edge_index)

        return F.log_softmax(x, dim=1)


from torch_geometric.nn import GATConv


class GATNet(torch.nn.Module):
    def __init__(self,input_features):
        super(GATNet,self).__init__()
        self.gat1=GATConv(input_features,16,16,dropout=0.6)
        self.gat2=GATConv(256,8,8,dropout=0.6)
        self.gat3=GATConv(64,4,4,dropout=0.6)
        self.gat3=GATConv(16,2,1,dropout=0.6)

    def forward(self,data):
        x,edge_index=data.x, data.edge_index
        x=self.gat1(x,edge_index)
        x=self.gat2(x,edge_index)
        x=self.gat3(x,edge_index)
        x=self.gat4(x,edge_index)
        return F.log_softmax(x,dim=1)
