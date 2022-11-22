from D_graph.graph_runner import GNN_runner
from data.mvtec3d import mvtec3d_classes
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gnn_runner = GNN_runner()
classes = mvtec3d_classes()
epochs = 5
cls = "tire"
for epoch in range(epochs):
    # cls = "tire"
    # gnn_runner = GNN_runner()
    # gnn_runner.fit(cls)
    for cls in classes:
        print(f"epoch: {epoch}")
        print(f"\nRunning on class {cls}\n")
        gnn_runner = gnn_runner.fit(cls)
    # gnn_runner.evaluate(cls)
        gnn_runner.evaluate(cls)
