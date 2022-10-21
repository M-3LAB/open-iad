from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from joblib import Parallel, delayed
import os
import numpy as np
import pandas as pd
from collections import Iterable

def flatten(items, ignore_types=(str, bytes)):
    for x in items:
        if isinstance(x, Iterable):
            yield from flatten(x)
        else:
            yield x

class Repeat(Dataset):
    def __init__(self, org_dataset, new_length):
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]


class MTD(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, size, require_mask=False, transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with the MTD dataset.
            class_name (string): class to load.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.require_mask = require_mask
        self.mode = mode
        self.size = size
        # find test images
        if self.mode == "train":
            self.image_names = list((self.root_dir / "train" / "good").glob("*.jpg"))
            print("loading MTD images")
            # during training we cache the smaller images for performance reasons (not a good coding style)
            self.imgs = (Parallel(n_jobs=10)(
                delayed(lambda file: Image.open(file).resize((size, size)).convert("RGB"))(file) for file in
                self.image_names))
            print(f"loaded training set : {len(self.imgs)} images")
        else:
            # test mode
            self.image_names = list((self.root_dir / "test").glob(str(Path("*") / "*.jpg")))
            self.gt_names = list((self.root_dir / "gt").glob(str(Path("*") / "*.png")))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            # img = Image.open(self.image_names[idx])
            # img = img.convert("RGB")
            img = self.imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img
        else:
            filename = self.image_names[idx]
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size, self.size)).convert("RGB")
            if self.require_mask:
                gtname = self.gt_names[idx]
                gt = Image.open(gtname).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                    gt = self.transform(gt)
                return img, gt, label != "good"
            else:
                if self.transform is not None:
                    img = self.transform(img)
                return img, label != "good"
