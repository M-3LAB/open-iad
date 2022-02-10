import torch
from torch.utils.data import Dataset
import os
from PIL import Image

__all__ = ['MTD', 'mtd_classes']

#TODO: Jinbao Not Finished yet
#
def mtd_classes():
    pass

class MTD(Dataset):
    def __init__(self, data_path, class_name, mode='centralized', phase='train', 
                 data_transform=None, mask_transform=None):

        self.data_path = data_path
        self.mode = mode
        self.phase = phase
        self.data_transform = data_transform 
        self.mask_transform = mask_transform
        self.class_name = class_name

        assert self.class_name in mtd_classes()
        # load dataset
        self.x, self.y. self.mask = self.load_dataset_folder()

        # data preprocessing 
        self.data_transform = data_transform
        self.mask_transformk = mask_transform

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        pass