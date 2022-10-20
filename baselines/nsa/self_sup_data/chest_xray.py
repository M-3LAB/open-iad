import os
import tarfile
from PIL import Image
from tqdm import tqdm
import urllib.request
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from .self_sup_tasks import patch_ex


class SelfSupChestXRay(Dataset):
    def __init__(self, data_dir, normal_files, mask_dir=None, mask_ids=[], anom_files=None, 
                 is_train=True, res=256, transform=None, self_sup_args={}):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.mask_ids = mask_ids
        self.is_train = is_train
        self.normal_files = normal_files
        self.anom_files = anom_files
        self.res = res

        # set transforms
        self.transform = transform
        self.final_transform = T.ToTensor()

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder(res)

        self.self_sup = is_train
        self.self_sup_args = self_sup_args
        self.prev_idx = np.random.randint(len(self.x))

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        if self.transform is not None:
            x = self.transform(x)
        x = np.asarray(x)[..., None]

        if self.self_sup:
            p = self.x[self.prev_idx] 
            if self.transform is not None:
                p = self.transform(p)
            p = np.asarray(p)[..., None]  
            x, mask = patch_ex(x, p, **self.self_sup_args)
            mask = torch.tensor(mask[None, ..., 0]).float()
            self.prev_idx = idx
        else:
            if mask is not None:
                if self.transform is not None:
                    mask = self.transform(mask)  # tranform should be deterministic when testing
                mask = self.final_transform(mask)
            else:
                mask = y * torch.ones((1, self.res, self.res))
        
        x = self.final_transform(x)
        return x, y, mask

    def __len__(self):
        return len(self.x)

    def configure_self_sup(self, on=True, self_sup_args={}):
        self.self_sup = on 
        self.self_sup_args.update(self_sup_args)

    def load_dataset_folder(self, res):
        transform = T.Resize(res, Image.ANTIALIAS)
        xs = []
        y = []
        for f in tqdm(self.normal_files, desc='read normal images'):
            xs.append(transform(Image.open(os.path.join(self.data_dir, f)).convert('L'))) 
            y.append(0)
        mask = [None for _ in range(len(xs))]
        if self.anom_files is not None:
            for f in tqdm(self.anom_files, desc='read anomaly images'):
                xs.append(transform(Image.open(os.path.join(self.data_dir, f)).convert('L'))) 
                y.append(1)
                if self.mask_dir is not None and f[:12] in self.mask_ids:
                    mask.append(transform(Image.open(os.path.join(self.mask_dir, f[:12] + '_bbox.png')).convert('L'))) 
                else:
                    mask.append(None) 
            mask += [None for _ in range(len(self.anom_files))]
        return list(xs), list(y), mask