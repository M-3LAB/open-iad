import torch
from torch.utils.data import Dataset
import os

__all__ = ['MVTecDataset']

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
    def __init__(self, data_path, mode='train', transform=None, mask_transform=None):
        self.data_path = data_path
        self.mode = mode

    def __getitem__(self):
        pass