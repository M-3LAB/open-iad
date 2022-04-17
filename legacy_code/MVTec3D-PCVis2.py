# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 21:02:28 2022

@author: LiuJiaqi
"""

import open3d as o3d
import numpy as np

path = 'test.xyzrgb'
pcd = o3d.io.read_point_cloud(path)
# point = np.asarray(pcd.points)  # 将点转换为numpy数组
# o3d.io.write_point_cloud('0000.pcd', pcd)

# path为文件路径
# pcd.paint_uniform_color([1, 0.706, 0])  # 修改点云颜色
# o3d.visualization.draw_geometries([pcd], zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])
o3d.visualization.draw_geometries([pcd])