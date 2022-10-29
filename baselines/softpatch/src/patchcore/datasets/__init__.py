import numpy as np
import random
import PIL
import torch
from torchvision import transforms


#添加椒盐噪声
class AddSaltPepperNoise(object):

    def __init__(self, density=0.0, p=0.5):
        self.density = density
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:  # 概率的判断
            img = np.array(img)  # 图片转numpy
            h, w, c = img.shape
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
            mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
            img[mask == 0] = 0  # 椒
            img[mask == 1] = 255  # 盐
            img = PIL.Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy转图片
            return img
        else:
            return img


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class NoiseDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            source,
    ):
        self.source = source
        self.transform_noise = transforms.Compose([
            # transforms.RandomChoice(transforms),
            # AddSaltPepperNoise(0.05, 1),
            # AddGaussianNoise(std=0.05),
            # transforms.GaussianBlur(3),
            # transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation(10),
            # transforms.RandomAffine(10, (0.1, 0.1), (0.9, 1.1), 10)
        ])


    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        item = self.source[idx]

        item["image"] = self.transform_noise(item["image"])
        return item