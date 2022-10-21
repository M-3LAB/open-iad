import torch
import torch.nn as nn
from torchvision import transforms

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