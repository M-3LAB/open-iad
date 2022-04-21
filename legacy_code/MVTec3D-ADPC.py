# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 16:02:10 2022

@author: LiuJiaqi
"""

from skimage import io

import numpy as np
 
# data=img.getdata()
  
xyz = io.imread('/disk4/xgy/mvtec/3D/rope/test/contamination/xyz/001.tiff')
rgb = io.imread('/disk4/xgy/mvtec/3D/rope/test/contamination/rgb/001.png')
AD = io.imread('/disk4/xgy/mvtec/3D/rope/test/contamination/gt/001.png')
pos = AD!=0
rgb[pos,0] = AD[pos]
rgb[pos,1:3] = 0
rgb = rgb/255
img = np.concatenate([xyz,rgb], axis=2)
img.resize(img.shape[0]*img.shape[1],6)

# AD.resize(AD.shape[0]*AD.shape[1],1)
# print(AD.shape)
np.savetxt('./legacy_code/test.xyzrgb', img, fmt='%f', delimiter=' ')