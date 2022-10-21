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


class MVTecMTDjoint(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mvtec_dir, mtd_dir, size, transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with the MTD dataset.
            class_name (string): class to load.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        mvtec_classes = ['leather', 'bottle', 'metal_nut',
                         'grid', 'screw', 'zipper',
                         'tile', 'hazelnut', 'toothbrush',
                         'wood', 'transistor', 'pill',
                         'carpet', 'capsule', 'cable']

        self.mvtec_dir = Path(mvtec_dir)
        self.mtd_dir = Path(mtd_dir)
        self.task_mvtec_classes = mvtec_classes
        self.transform = transform
        self.mode = mode
        self.size = size
        self.all_imgs = []
        self.all_image_names = []
        # find test images
        if self.mode == "train":
            self.mtd_image_names = list((self.mtd_dir / "train" / "good").glob("*.jpg"))
            self.all_image_names.append(self.mtd_image_names)
            print("loading MTD images")
            # during training we cache the smaller images for performance reasons (not a good coding style)
            self.mtd_imgs = (Parallel(n_jobs=10)(
                delayed(lambda file: Image.open(file).resize((size, size)).convert("RGB"))(file) for file in
                self.mtd_image_names))
            self.all_imgs.append(self.mtd_imgs)
            print(f"loaded MTD : {len(self.mtd_imgs)} images")

            for class_name in self.task_mvtec_classes:
                self.mvtec_image_names = list((self.mvtec_dir / class_name / "train" / "good").glob("*.png"))
                self.all_image_names.append(self.mvtec_image_names)
                print("loading MVTec images")
                # during training we cache the smaller images for performance reasons (not a good coding style)
                self.mvtec_imgs = (Parallel(n_jobs=10)(
                    delayed(lambda file: Image.open(file).resize((size, size)).convert("RGB"))(file) for file in
                    self.mvtec_image_names))
                self.all_imgs.append(self.mvtec_imgs)
                print(f"loaded {class_name} : {len(self.mvtec_imgs)} images")
        else:
            # test mode
            self.mtd_image_names = list((self.mtd_dir / "test").glob(str(Path("*") / "*.jpg")))
            self.mtd_gt_names = list((self.mtd_dir / "gt").glob(str(Path("*") / "*.png")))
            for class_name in self.task_mvtec_classes:
                self.mvtec_image_names = list((self.mvtec_dir / class_name / "test").glob(str(Path("*") / "*.png")))
                self.all_image_names.append(self.mvtec_image_names)
        self.all_imgs, self.all_image_names = list(flatten(self.all_imgs)), list(flatten(self.all_image_names))


    def __len__(self):
        return len(self.all_image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            # img = Image.open(self.image_names[idx])
            # img = img.convert("RGB")
            img = self.all_imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img
        else:
            filename = self.all_image_names[idx]
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size, self.size)).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, label != "good"
