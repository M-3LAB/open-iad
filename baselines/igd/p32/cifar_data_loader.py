import os
import random
import torch.utils.data

from PIL import Image


class MyDataLoader(torch.utils.data.Dataset):

    # constructor of the class
    def __init__(self, path, transform, normal_number=0, shuffle=False):
        self.current_normal_number = normal_number
        self.transform = transform
        images = [os.path.join(path, img) for img in os.listdir(path)]
        if shuffle:
            random.shuffle(images)
        self.images = images

    def __getitem__(self, index):
        image_path = self.images[index]
        label = int(image_path.split('/')[-1].split('_')[0])
        data = Image.open(image_path)
        data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.images)
