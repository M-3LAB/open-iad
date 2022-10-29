import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)
        if self.is_train:
            return image
        else:
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("/test/", "/ground_truth/").replace(
                        ".png", "_mask.png"
                    )
                )
                target = self.target_transform(target)
            return image, target

    def __len__(self):
        return len(self.image_files)
