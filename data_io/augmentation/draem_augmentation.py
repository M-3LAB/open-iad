from asyncore import read
from ctypes import resize
from locale import normalize
from torchvision import transforms as T
from PIL import Image
import torch
import tifffile
import numpy as np
import cv2
import imgaug.augmenters as iaa
import glob
from data_io.augmentation.perlin import rand_perlin_2d_np
import os

__all__ =  ['mvtec_2d_resize', 'mvtec_2d_image_transform', 'mvtec_2d_mask_transform', 
            'aug_draem_3d_train', 'aug_draem_3d_test', 'ImageToPatch']

# 2D 

mvtec_2d_resize = T.Compose([T.Resize(size=1000)])

mvtec_2d_image_transform = T.Compose([T.Resize(224),
                                      T.CenterCrop(224),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                                      ])

mvtec_2d_mask_transform = T.Compose([T.Resize(224),
                                     T.CenterCrop(224),
                                     T.ToTensor()
                                     ])

augmenters_DREAM = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                    iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                    iaa.pillike.EnhanceSharpness(),
                    iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                    iaa.Solarize(0.5, threshold=(32,128)),
                    iaa.Posterize(),
                    iaa.Invert(),
                    iaa.pillike.Autocontrast(),
                    iaa.pillike.Equalize(),
                    iaa.Affine(rotate=(-45, 45))
                ]

rot_DREAM = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

def read_tiff(tiff):
    # tiff_img: numpy format
    tiff_img = tifffile.imread(tiff)
    return tiff_img

def np_to_torch(array):
    array = np.transpose(array, (2, 0, 1)) 
    tensor = torch.from_numpy(array)
    return tensor
    

def tiff_to_depth_numpy(tiff, resized_img_size=256, duplicate=3, normalize=False):
    depth_map = np.array(tiff[:, :, 2]).astype(np.float32)
    depth_map = np.resize(depth_map, (resized_img_size, resized_img_size)) 

    if normalize:
        depth_map = min_max_normlize(depth_map)

    depth_map = np.expand_dims(depth_map, axis=2)
    if duplicate == 3:
        depth_map = np.repeat(depth_map, repeats=3, axis=2)
    return depth_map


def tiff_to_depth_torch(tiff, resized_img_size=256, duplicate=1):
    depth_map = np.array(tiff[:, :, 2])
    # Duplicate depth_map into 3 channels, Convert numpy format into BCHW
    if duplicate == 3: 
        depth_map = np.repeat(depth_map[:, :, np.newaxis], 3, axis=2)
        depth_map = torch.from_numpy(depth_map).permute(2, 0, 1).unsqueeze(dim=0)
    else: # duplicate = 1 
        # One channel, Convert numpy into BCHW
        depth_map = torch.from_numpy(depth_map).unsqueeze(dim=0).unsqueeze(dim=0)

    # Downsampling, Nearest Interpolation
    resized_depth_map = torch.nn.functional.interpolate(depth_map, size=(resized_img_size[0], resized_img_size[1]),
                                                        mode='nearest')
    return resized_depth_map

def sun3d_get_depth_image_list(data_path):
    xtion_depth_path = os.path.join(data_path, 'xtion/sun3ddata/*/*/*/depth/*.png') 
    other_depth_path = os.path.join(data_path, '*/*/*/depth/*.png')

    depth_list_sun3d = glob.glob(xtion_depth_path)
    depth_list_other = glob.glob(other_depth_path)

    depth_list = depth_list_sun3d + depth_list_other
    return depth_list

def sun3d_get_rgb_image_list(data_path):
    xtion_img_path = os.path.join(data_path, 'xtion/sun3ddata/*/*/*/image/*.jpg')
    other_img_path = os.path.join(data_path, '*/*/*/image/*.jpg')

    image_list_sun3d = glob.glob(xtion_img_path)
    image_list_other = glob.glob(other_img_path)

    image_list = image_list_sun3d + image_list_other
    return image_list

def randAugmenter():
    aug_ind = np.random.choice(np.arange(len(augmenters_DREAM)), 3, replace=False)
    aug = iaa.Sequential([augmenters_DREAM[aug_ind[0]],
                          augmenters_DREAM[aug_ind[1]],
                          augmenters_DREAM[aug_ind[2]]]
                        )
    return aug

def min_max_normlize(numpy):
    numpy_norm = (numpy - np.min(numpy)) / (np.max(numpy) - np.min(numpy) + 1e-8)
    return numpy_norm

def augment_image_DREAM(image, depth, extra_anomaly_rgb, extra_anomaly_depth, resize_shape=[256,256]):
    
    aug = randAugmenter()
    perlin_scale = 6
    min_perlin_scale = 0
    anomaly_source_img = cv2.imread(extra_anomaly_rgb)
    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(resize_shape[1], resize_shape[0]))
    anomaly_source_depth = cv2.imread(extra_anomaly_depth)
    anomaly_source_depth = cv2.resize(anomaly_source_depth, dsize=(resize_shape[1], resize_shape[0]))

    anomaly_img_augmented = aug(image=anomaly_source_img)
    anomaly_depth_augmented = aug(image=anomaly_source_depth)
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

    perlin_noise = rand_perlin_2d_np((resize_shape[0], resize_shape[1]), (perlin_scalex, perlin_scaley))
    perlin_noise = rot_DREAM(image=perlin_noise)
    threshold = 0.5
    perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
    perlin_thr = np.expand_dims(perlin_thr, axis=2)

    img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
    depth_thr = anomaly_depth_augmented.astype(np.float32) * perlin_thr / 255.0

    beta = torch.rand(1).numpy()[0] * 0.8

    augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
        perlin_thr)
    augmented_depth = depth * (1 - perlin_thr) + (1 - beta) * depth_thr + beta * depth * (
        perlin_thr)

    no_anomaly = torch.rand(1).numpy()[0]
    if no_anomaly > 0.5:
        image = image.astype(np.float32)
        depth = depth.astype(np.float32)
        return image, depth, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
    else:
        augmented_image = augmented_image.astype(np.float32)
        augmented_depth = augmented_depth.astype(np.float32)
        msk = (perlin_thr).astype(np.float32)
        augmented_image = msk * augmented_image + (1-msk)*image
        augmented_depth = msk * augmented_depth + (1-msk)*depth
        has_anomaly = 1.0
        if np.sum(msk) == 0:
            has_anomaly=0.0
        return augmented_image, augmented_depth, msk, np.array([has_anomaly],dtype=np.float32)

def aug_draem_3d_train(image_path, tiff_path, extra_rgbd_path, resize_shape=[256,256], depth_duplicate=3):
    rgb_anomaly_source_list = sun3d_get_rgb_image_list(extra_rgbd_path)
    depth_anomaly_source_list = sun3d_get_depth_image_list(extra_rgbd_path)
    anomaly_source_idx = torch.randint(0, len(rgb_anomaly_source_list), (1,)).item()
    
    img = cv2.imread(image_path)
    img = cv2.resize(img, dsize=(resize_shape[1], resize_shape[0]))
    tiff_img = read_tiff(tiff_path)
    tiff_img = cv2.resize(tiff_img, dsize=(resize_shape[1], resize_shape[0]))

    do_aug_orig = torch.rand(1).numpy()[0] > 0.7
    if do_aug_orig:
        img = rot_DREAM(image=img)
        tiff_img = rot_DREAM(image=tiff_img)

    img_np = np.array(img).reshape((img.shape[0], img.shape[1], img.shape[2])).astype(np.float32)
    img_np = min_max_normlize(img_np)

    depth = tiff_to_depth_numpy(tiff=tiff_img, resized_img_size=resize_shape[0], duplicate=depth_duplicate)
    depth = min_max_normlize(depth)

    augmented_img, augmented_depth, anomaly_mask, has_anomaly = augment_image_DREAM(img_np, depth, 
                                                                                    extra_anomaly_rgb=rgb_anomaly_source_list[anomaly_source_idx],
                                                                                    extra_anomaly_depth=depth_anomaly_source_list[anomaly_source_idx])

    img_torch = np_to_torch(img_np)
    augmented_img_torch = np_to_torch(augmented_img) 
    depth_torch = np_to_torch(depth)
    augmented_depth_torch = np_to_torch(augmented_depth)
    mask_torch = np_to_torch(anomaly_mask) 
    anomaly_label_torch = np_to_torch(has_anomaly)

    return img_torch, augmented_img_torch, depth_torch, augmented_depth_torch, mask_torch, anomaly_label_torch

def transform_image_DREAM_noperlin(image, resize_shape=[256,256]):
    if resize_shape != None:
        image = cv2.resize(image, dsize=(resize_shape[1], resize_shape[0]))
    image = min_max_normlize(image)
    image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)


def transform_depth_DREAM_noperlin(tiff_img, resize_shape=[256,256], depth_duplicate=1):

    depth = tiff_to_depth_numpy(tiff_img, resized_img_size=resize_shape[0], 
                                duplicate=depth_duplicate, normalize=True)

    depth_torch = np_to_torch(depth)
    return depth_torch
    

def aug_draem_3d_test(img_path, tiff_path, mask_path, depth_duplicate, resize_shape=[256, 256]):

    raw_img = cv2.imread(img_path)
    raw_img = transform_image_DREAM_noperlin(raw_img)
    tiff_img = read_tiff(tiff_path)
    depth_map = transform_depth_DREAM_noperlin(tiff_img, resize_shape=resize_shape, depth_duplicate=depth_duplicate)

    if label == 0:
        label = np.array([0], dtype=np.float32)
        mask = torch.zeros([1, raw_img.shape[1], raw_img.shape[2]]) 
    else: 
        label = np.array([1], dtype=np.float32)
        mask = transform_image_DREAM_noperlin(mask_path)

    label = torch.from_numpy(label)

    return raw_img, depth_map, mask, label 

#TODO: Tile, Jingbao
class ImageToPatch:
    def __init__(self):
        pass
