import numpy as np
from timm.models.layers import to_2tuple
import torch
from PIL import Image
from torchvision import transforms
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, \
    IMAGENET_INCEPTION_STD
from torchvision.transforms import ToPILImage


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask  # [196]


class MaskImg:
    def __init__(self, device, size, mask_ratio, patch_size, colorJitter=0.1, transform=None):
        self.device = device
        self.input_size = size
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        # self.patch_size = to_2tuple(patch_size)
        self.window_size = (self.input_size // self.patch_size[0], self.input_size // self.patch_size[1])
        self.masked_position_generator = RandomMaskingGenerator(self.window_size, self.mask_ratio)

        self.transform = transform
        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(brightness=colorJitter,
                                                      contrast=colorJitter,
                                                      saturation=colorJitter,
                                                      hue=colorJitter)

    def __call__(self, img):
        oriimg = self.transform(img)

        bool_masked_pos = self.masked_position_generator()
        bool_masked_pos = torch.from_numpy(bool_masked_pos)

        # imagenet_default_mean_and_std = True
        # mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        #
        # # transform = transforms.Compose([
        # #     # transforms.RandomResizedCrop(input_size),
        # #     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        # #     transforms.ToTensor(),
        # #     transforms.Normalize(
        # #         mean=torch.tensor(mean),
        # #         std=torch.tensor(std))
        # # ])
        img = self.colorJitter(img)
        img = self.transform(img)
        img = img[None, :]
        bool_masked_pos = bool_masked_pos[None, :]
        # img = img.to(self.device)
        bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)

        # save original img
        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN)[None, :, None, None]
        std = torch.as_tensor(IMAGENET_DEFAULT_STD)[None, :, None, None]
        ori_img = img * std + mean  # in [0, 1]

        img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=self.patch_size[0], p2=self.patch_size[1])
        img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (
                img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')

        # make mask
        mask = torch.ones_like(img_patch)
        mask[bool_masked_pos] = 0
        mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
        h = int(self.input_size / self.patch_size[0])
        w = int(self.input_size / self.patch_size[1])
        mask = rearrange(mask, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=self.patch_size[0], p2=self.patch_size[1], h=h, w=w)

        # save reconstruction img
        rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
        # Notice: To visualize the reconstruction image, we add the predict and the original mean and var of each patch. Issue #40
        rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(
            dim=-2,
            keepdim=True)
        rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=self.patch_size[0], p2=self.patch_size[1], h=h,
                            w=w)
        # img = ToPILImage()(rec_img[0, :].clip(0, 0.996))

        # save random mask img
        img_mask = rec_img * mask
        img = img_mask[0, :]
        # img = ToPILImage()(img_mask[0, :])

        return oriimg, img




