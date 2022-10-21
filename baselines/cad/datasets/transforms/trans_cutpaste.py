import random
import math
from torchvision import transforms
import torch
from PIL import ImageFilter, Image
import numpy as np


def cut_paste_collate_fn(batch):
    # cutPaste return 2 tuples of tuples we convert them into a list of tuples
    img_types = list(zip(*batch))
    #     print(list(zip(*batch)))
    return [torch.stack(imgs) for imgs in img_types]


class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)

class CutPaste1(object):
    """Base class for both cutpaste variants with common operations"""

    def __init__(self, colorJitter=0.1, transform=None):
        self.transform = transform
        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(brightness=colorJitter,
                                                      contrast=colorJitter,
                                                      saturation=colorJitter,
                                                      hue=colorJitter)

    def __call__(self, img):
        # apply transforms to both images
        if self.transform:
            org_img = self.transform(img)
            if self.colorJitter:
                img = self.colorJitter(img)
            img = self.transform(img)
        return org_img, img

class CutPaste(object):
    """Base class for both cutpaste variants with common operations"""

    def __init__(self, colorJitter=0.1, transform=None):
        self.transform = transform

        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(brightness=colorJitter,
                                                      contrast=colorJitter,
                                                      saturation=colorJitter,
                                                      hue=colorJitter)
        # self.k = random.randint(0,3)

    def __call__(self, org_img, img):
        # apply transforms to both images
        if self.transform:
            img = self.transform(img)
            # org_img = np.rot90(org_img, k=self.k)
            # org_img = Image.fromarray(org_img)
            org_img = self.transform(org_img)
        return org_img, img


class CutPasteNormal(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """

    def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3, **kwags):
        super(CutPasteNormal, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
        # TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        h = img.size[0]
        w = img.size[1]

        new_aspect_ratio = random.uniform(0.1, 1)

        augmented = img.copy()
        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h

        # sample in log space
        log_ratio = torch.log(torch.tensor((new_aspect_ratio, 1 / new_aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))

        # one might also want to sample from other images. currently we only sample from the image itself
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)

        # Before pasting, we apply color jitter.we rotate or jitter pixel values in the patch
        if self.colorJitter:
            patch = self.colorJitter(patch)

        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))

        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]

        augmented.paste(patch, insert_box)  # The pasted image patch always origins from the same image it is pasted to

        return super().__call__(img, augmented)


class BlurCutPasteNormal(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """

    def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3, **kwags):
        super(BlurCutPasteNormal, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
        # TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        h = img.size[0]
        w = img.size[1]

        augmented = img.copy()
        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h

        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1 / self.aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))

        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))

        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]

        augmented = augmented.filter(MyGaussianBlur(radius=5, bounds=insert_box))
        return super().__call__(img, augmented)


class MultiCutPasteNormal(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """

    def __init__(self, area_ratio=[0.002, 0.015], aspect_ratio=0.3, **kwags):
        super(MultiCutPasteNormal, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
        # TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        h = img.size[0]
        w = img.size[1]

        left, right = random.uniform(0.001, 0.003), random.uniform(0.01, 0.02)
        new_area_ratio = [left, right]
        new_aspect_ratio = random.uniform(0.1, 1)

        augmented = img.copy()

        num_patch = random.randint(1, 5)
        for _ in range(num_patch):
            # ratio between area_ratio[0] and area_ratio[1]
            ratio_area = random.uniform(new_area_ratio[0], new_area_ratio[1]) * w * h

            # sample in log space
            log_ratio = torch.log(torch.tensor((new_aspect_ratio, 1 / new_aspect_ratio)))
            aspect = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            cut_w = int(round(math.sqrt(ratio_area * aspect)))
            cut_h = int(round(math.sqrt(ratio_area / aspect)))

            # one might also want to sample from other images. currently we only sample from the image itself
            from_location_h = int(random.uniform(0, h - cut_h))
            from_location_w = int(random.uniform(0, w - cut_w))

            box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
            patch = img.crop(box)

            # Before pasting, we apply color jitter.we rotate or jitter pixel values in the patch
            if self.colorJitter:
                patch = self.colorJitter(patch)

            to_location_h = int(random.uniform(0, h - cut_h))
            to_location_w = int(random.uniform(0, w - cut_w))

            insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]

            augmented.paste(patch,
                            insert_box)  # The pasted image patch always origins from the same image it is pasted to

        return super().__call__(img, augmented)


class SwapPatch(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """

    def __init__(self, area_ratio=[0.0002, 0.0015], aspect_ratio=0.3, **kwags):
        super(SwapPatch, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
        # TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        h = img.size[0]
        w = img.size[1]

        augmented = img.copy()
        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h

        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1 / self.aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))

        # one might also want to sample from other images. currently we only sample from the image itself
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box1 = [0, from_location_h, w, from_location_h + cut_h]
        patch1 = img.crop(box1)

        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))

        box2 = [0, to_location_h, w, to_location_h + cut_h]
        patch2 = img.crop(box2)

        augmented.paste(patch1, box2)  # The pasted image patch always origins from the same image it is pasted to
        augmented.paste(patch2, box1)

        return super().__call__(img, augmented)


class CutPasteScar(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        width (list): width to sample from. List of [min, max]
        height (list): height to sample from. List of [min, max]
        rotation (list): rotation to sample from. List of [min, max]
    """

    def __init__(self, width=[2, 16], height=[10, 25], rotation=[-45, 45], **kwags):
        super(CutPasteScar, self).__init__(**kwags)
        self.width = width
        self.height = height
        self.rotation = rotation

    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]

        # cut region
        cut_w = random.uniform(*self.width)
        cut_h = random.uniform(*self.height)

        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)

        if self.colorJitter:
            patch = self.colorJitter(patch)

        # rotate
        rot_deg = random.uniform(*self.rotation)
        patch = patch.convert("RGBA").rotate(rot_deg, expand=True)

        # paste
        to_location_h = int(random.uniform(0, h - patch.size[0]))
        to_location_w = int(random.uniform(0, w - patch.size[1]))

        mask = patch.split()[-1]
        patch = patch.convert("RGB")

        augmented = img.copy()
        augmented.paste(patch, (to_location_w, to_location_h), mask=mask)

        return super().__call__(img, augmented)


class MultiCutPasteScar(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        width (list): width to sample from. List of [min, max]
        height (list): height to sample from. List of [min, max]
        rotation (list): rotation to sample from. List of [min, max]
    """

    def __init__(self, width=[2, 16], height=[10, 25], rotation=[-45, 45], **kwags):
        super(MultiCutPasteScar, self).__init__(**kwags)
        self.width = width
        self.height = height
        self.rotation = rotation

    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]
        augmented = img.copy()

        num_patch = random.randint(1, 5)
        for _ in range(num_patch):
            # cut region
            cut_w = random.uniform(*self.width)
            cut_h = random.uniform(*self.height)

            from_location_h = int(random.uniform(0, h - cut_h))
            from_location_w = int(random.uniform(0, w - cut_w))

            box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
            patch = img.crop(box)

            if self.colorJitter:
                patch = self.colorJitter(patch)

            # rotate
            rot_deg = random.uniform(*self.rotation)
            patch = patch.convert("RGBA").rotate(rot_deg, expand=True)

            # paste
            to_location_h = int(random.uniform(0, h - patch.size[0]))
            to_location_w = int(random.uniform(0, w - patch.size[1]))

            mask = patch.split()[-1]
            patch = patch.convert("RGB")

            augmented.paste(patch, (to_location_w, to_location_h), mask=mask)

        return super().__call__(img, augmented)


class CutPasteUnion(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)

    def __call__(self, img):
        r = random.uniform(0, 1)
        if r < 0.5:
            return self.normal(img)
        else:
            return self.scar(img)


class CutPaste3Way(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)

    def __call__(self, img):
        org, cutpaste_normal = self.normal(img)
        _, cutpaste_scar = self.scar(img)

        return org, cutpaste_normal, cutpaste_scar

