import torch
import os
import tifffile as tiff
from data_io.mvtec3d import *
import numpy as np

tiff_path = 'test_data/good/000.tiff'

tiff_img = read_tiff(tiff_path)
print(tiff_img.shape)
print(tiff_img[:,:, 2])
tiff_img = torch.from_numpy(tiff_img).permute(2, 0, 1)
print(tiff_img.size())

depth_map = tiff_to_depth(tiff_img)
print(depth_map.shape)

depth_map_3channel = np.repeat(depth_map[:, :, np.newaxis], 3, axis=2)
print(depth_map_3channel.shape)
