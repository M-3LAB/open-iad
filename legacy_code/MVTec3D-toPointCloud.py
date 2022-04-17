# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 13:31:49 2022

@author: LiuJiaqi
"""

from skimage import io

import numpy as np
 
# data=img.getdata()
  
xyz = io.imread('001.tiff')
rgb = io.imread('001.png')
rgb = rgb/255
img = np.concatenate([xyz,rgb], axis=2)
img.resize(img.shape[0]*img.shape[1],6)
np.savetxt('test.xyzrgb', img, fmt='%f', delimiter=' ')