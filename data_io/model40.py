import torch
import torch.nn as nn
from torch.utils.data import Dataset

__all__ = ['Model40']

class Model40(Dataset):
    def __init__(self):
        super(Model40).__init__()
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        pass