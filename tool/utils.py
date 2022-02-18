import os
from PIL import Image
import torch
import cv2
import random
import numpy as np

__all__ = ['to_batch', 'seed_everything']

def to_batch(images, transforms, device):
    """
    Convert a list of numpy array images to a pytorch tensor batch with given transforms.
    Args:
        images: List, np.ndarray
        transforms: torchvision T.Compose

    return torch.tensor
    """
    assert len(images) > 0

    transformed_images = []
    for i, image in enumerate(images):
        image = Image.fromarray(image).convert('RGB')
        transformed_images.append(transforms(image))

    height, width = transformed_images[0].shape[1:3]
    batch = torch.zeros((len(images), 3, height, width))

    for i, transformed_image in enumerate(transformed_images):
        batch[i] = transformed_image

    return batch.to(device)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

