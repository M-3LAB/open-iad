# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:02:02 2022

@author: LiuJiaqi
"""

import open3d as o3d
import numpy as np

txt_path = 'test.xyzrgb'
# 通过numpy读取txt点云
pcd = np.genfromtxt(txt_path, delimiter=" ")

pcd_vector = o3d.geometry.PointCloud()
# 加载点坐标
pcd_vector.points = o3d.utility.Vector3dVector(pcd[:, :3])
# o3d.visualization.draw_geometries([pcd_vector])

# pcd_vector.normals = o3d.utility.Vector3dVector(pcd[:, 3:6])
pcd_vector.colors = o3d.utility.Vector3dVector(pcd[:, 3:6])
o3d.visualization.draw_geometries([pcd_vector])