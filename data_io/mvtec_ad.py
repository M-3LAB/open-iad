import torch
from torch.utils.data import Dataset
import os
from PIL import Image

__all__ = ['MVTec2D', 'MVTec3D']

def mvtec_classes():
    return [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

class MVTec2D(Dataset):
    def __init__(self, data_path, class_name, mode='train', data_transform=None, 
                 mask_transform=None):

        self.data_path = data_path
        self.mode = mode
        self.data_transform = data_transform 
        self.mask_transform = mask_transform
        self.class_name = class_name

        assert self.class_name in mvtec_classes


    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def load_dataset_folder(self):
        x, y, mask = [], [], []
        img_dir = os.path.join(self.data_path, self.class_name, self.mode)
        gt_dir = os.path.join(self.data_path, self.class_name, self.mode)


class MVTec3D(Dataset):
    def __init__(self):
        pass

    def __getitem__(self):
        pass