# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:33:52 2022

@author: v_jiaqiiliu
"""

from skimage import io

import numpy as np

import torch
from torch.nn import functional as F
 
# data=img.getdata()

def bilinearlyDownsample(image, out_h, out_w):
    # image is 3D
    # grid is like mesh_grid
    image = image.unsqueeze(0)
    new_h = torch.linspace(-1, 1, out_h).view(-1, 1).repeat(1, out_w)
    new_w = torch.linspace(-1, 1, out_w).repeat(out_h, 1)

    grid = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2)
    grid = grid.unsqueeze(0)
    # grid = grid.cuda()
    I = F.grid_sample(image, grid=grid, mode='bilinear', align_corners=True)
    I = I.squeeze(0) # remove the fake batch dimension
    I = I.permute(0,2,1)
    I = torch.flatten(I, start_dim=1, end_dim=2)
    return I
  
xyz = io.imread('001.tiff')
rgb = io.imread('001.png')
rgb = rgb/255
img = np.concatenate([xyz,rgb], axis=2)
# img = torch.from_numpy(img)
# img = img.permute(2,0,1)
# # img.to('cuda:0')
# img = img.to(torch.float32)


# z = img[2,:,:]
# avg = np.mean(z)
# pos = img[2,:,:]<avg
# print(pos.shape)
# # img = img[,pos]

# print(z.shape)

# print(avg)
# print(img.shape)

# maxz = torch.max(z)

# img[2,pos] = maxz

# img = bilinearlyDownsample(img, 100, 100)
# img = img.permute(1,0)
# img = img.numpy()

# # print(min(img[2,:]))
# # print(max(img[2,:]))
# # print(sum(img[2,:])/640000)

# temp = img.copy()
img.resize(img.shape[0]*img.shape[1],6)
avg = np.mean(img[:,2])


pos = img[:,2]>avg
img=img[pos]
print(img.shape)
np.savetxt('test.xyzrgb', img, fmt='%f', delimiter=' ')