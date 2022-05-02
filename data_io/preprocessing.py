import tifffile as tiff
import torch
import os
import open3d as o3d
import math
import argparse
from PIL import Image

"""
pc: Point Cloud
"""

def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])
