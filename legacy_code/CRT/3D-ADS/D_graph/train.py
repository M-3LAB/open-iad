from D_graph.graph_3d_runner import GNN_runner
from data.mvtec3d import mvtec3d_classes
import os
import torch

import numpy as np
from numpy import random
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
gnn_runner = GNN_runner()
classes = mvtec3d_classes()
epochs = 100


# cls = "cookie"
# for i in range(epochs):
#         print(i)
#         gnn_runner = gnn_runner.fit(cls)
#         print()
#         print()
#         print()
#         print()
#         print()
#         print()
#         print(i)
#         print("**************************************************************************************")
# a = 0.9+torch.rand(10)*0.2
# print(a*a)
# # select by mask
# x = torch.randn(3,4)
# mask = x.ge(0.5)
# print(x)
# print(mask)
# print(torch.masked_select(x,mask))
# x = torch.ones(5,4)
# x_data = x.mul(0.9+torch.rand(x.shape[0],x.shape[1])*0.2)
# print(x)
# print(x_data)
# y_data = [1 if random.random()>0.5 else 0 for i in range(x.shape[0])]
# print(y_data)
# mask = torch.tensor(y_data).view(-1,1)
# mask = mask>0

# mask = mask.repeat(1, x.shape[1])
# print(mask)

# a = x_data.masked_scatter_(mask,x) 

# print(a)







# a = [1 if random.random()>0.1 else 0 for i in range(10)]
# b = [1 if i==0 else 0 for i in a]
# print(a)
# print(b)
# pixel_auroc, img_auroc, pixel_ap, img_ap,pixel_pro = gnn_runner.evaluate(cls)
# print("pixel_auroc:",pixel_auroc)
# print("pixel_ap:",pixel_ap)
# print("img_auroc:",img_auroc)
# print("img_ap:",img_ap)
# print("pixel_pro:",pixel_pro)

# for epoch in range(epochs):
#     print(f"epoch: {epoch}")
#     print(f"\nRunning on class {cls}\n")
#     gnn_runner = gnn_runner.fit(cls)

# pixel_auroc, img_auroc, pixel_ap, img_ap = gnn_runner.evaluate(cls)
# print("pixel_auroc:",pixel_auroc)
# print("pixel_ap:",pixel_ap)
# print("img_auroc:",img_auroc)
# print("img_ap:",img_ap)












# for epoch in range(epochs):
#     for cls in classes:
#         print(f"epoch: {epoch}")
#         print(f"\nRunning on class {cls}\n")
#         gnn_runner = gnn_runner.fit(cls)

for cls in classes:
    print("eval:")
    print(f"\nRunning on class {cls}\n")
    pixel_auroc, img_auroc, pixel_ap, img_ap,pixel_pro = gnn_runner.evaluate(cls)
    print("pixel_auroc:",pixel_auroc)
    print("pixel_ap:",pixel_ap)     
    print("img_auroc:",img_auroc)
    print("img_ap:",img_ap)
    print("pixel_pro:",pixel_pro)



        # gnn_runner.eval(cls)
#         gnn_runner = gnn_runner.fit(cls)
#     print(f"epoch: {epoch}")
#     gnn_runner = gnn_runner.fit(cls)

# gnn_runner.eval(cls)


# pixel_auroc, img_auroc, pixel_ap, img_ap = gnn_runner.evaluate(cls)
# print(pixel_auroc)
# print(img_auroc)
# print(pixel_ap)
# print(img_ap)
# for epoch in range(epochs):
#     # cls = "tire"
#     # gnn_runner = GNN_runner()
#     # gnn_runner.fit(cls)
#     for cls in classes:
#         print(f"epoch: {epoch}")
#         print(f"\nRunning on class {cls}\n")
#         gnn_runner = gnn_runner.fit(cls)
#     # gnn_runner.evaluate(cls)
#         gnn_runner.evaluate(cls)
# from pygod.models import CONAD
# model = CONAD()
# print(model)