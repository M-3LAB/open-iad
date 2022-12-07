import torch
from pygod.utils import load_data
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import numpy as np
dataset = TUDataset(root='/ssd2/m3lab/usrs/crt/GCN/temp', name ='ENZYMES',use_node_attr=True)
loader = DataLoader(dataset,batch_size=4,shuffle=True)
# for data in loader:
#     print(data)

graph = load_data('inj_cora')
graph.train_mask = []
graph.val_mask = []
graph.test_mask = []
# graph = dataset
print(graph)
print(graph.y)

# print(graph.edge_index)
# print(graph.x)
# print(graph.y)
# print(graph.train_mask)
from pygod.models import DOMINANT 
from pygod.models import CONAD

from sklearn.metrics import roc_auc_score, average_precision_score 

def train_anomaly_detector(model,graph):
    # for batch in loader:
    #     batch.y = torch.zeros(batch.x.shape[0])
    #     batch.y = batch.y.bool()
    #     model = model.fit(batch)
    return model.fit(graph)

def eval_anomaly_detector(model, graph): 

    outlier_scores = model.decision_function(graph) 
    auc = roc_auc_score(graph.y.numpy(), outlier_scores) 
    ap = average_precision_score(graph.y.numpy(), outlier_scores) 
    print(outlier_scores)
    print(f'AUC Score: {auc:.3f}') 
    print(f'AP Score: {ap:.3f}') 


# graph.y = graph.y.bool()
# y = graph.y.numpy()
# print(np.where(y==1))

# model = DOMINANT(epoch=5,verbose=True)
# model = CONAD(epoch=5,verbose=True)
# model = train_anomaly_detector(model, graph)
# eval_anomaly_detector(model, graph)

# model.train_first = False
# for i in range(20):
#     model = train_anomaly_detector(model, graph)
#     print(model.train_first)
#     eval_anomaly_detector(model, graph)

from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2, 2,0],
                           [1, 0, 2, 1, 0,2]], dtype=torch.long)
x = torch.tensor([[10,0], [21,3], [32,5]], dtype=torch.float)
y = torch.tensor([[0], [1], [0]], dtype=torch.float)

graph = Data(x=x, edge_index=edge_index,y=y)
print(graph)
graph.y = graph.y.bool()
print(graph.y)
model = CONAD(epoch=1,verbose=True)
model = train_anomaly_detector(model, graph)
y_pre = model.decision_function(graph)
print(model.decision_function(graph))
eval_anomaly_detector(model, graph)

print(roc_auc_score([10,1,10], [1,0,1]))
print(average_precision_score([10,1,10], [1,0,1]))
print(average_precision_score([10,1,10], [0,1,0]))
