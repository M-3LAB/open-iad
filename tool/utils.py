import os
from PIL import Image
import torch

__all__ = ['to_batch']

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