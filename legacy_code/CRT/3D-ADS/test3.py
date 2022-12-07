# import torch
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# x = torch.randn(1, 3, 224, 224)
# # output = torch.nn.functional.unfold(x, kernel_size=(2, 3))
# print(x)
# print(x.size())
# x1 = x[:,:,0::4,0::4]
# x2 = x[:,:,1::4,1::4]
# x3 = x[:,:,2::4,2::4]
# x4 = x[:,:,3::4,3::4]
# print(x1.size())
# print(x2.size())
# print(x3.size())
# print(x4.size())

# output = torch.cat([x1,x2,x3,x4],dim=1)
# output = torch.nn.functional.conv2d(x,kernel_size=4,stride=4)
# print(output.size())
import numpy as np
a = []
print(16%3)
print(1 and 0)
index = 56
for i in range(index):
    for j in range(index):
        if (i>0):
            a.append([i*index+j,(i-1)*index+j])
        if (i<index-1):
            a.append([i*index+j,(i+1)*index+j])
        if (j>0):
            a.append([i*index+j,i*index+j-1])
        if (j<index-1):
            a.append([i*index+j,i*index+j+1])


print(a)

