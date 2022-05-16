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

__all__ =  ['mvtec_2d_resize', 'mvtec_2d_image_transform', 'mvtec_2d_mask_transform', 'aug_DREAM_3D']

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

def read_tiff(tiff):
    # tiff_img: numpy format
    tiff_img = tifffile.imread(tiff)
    return tiff_img

def tiff_to_depth(tiff, resized_img_size=256, duplicate=False):
    depth_map = np.array(tiff[:, :, 2])
    # Duplicate depth_map into 3 channels, Convert numpy format into BCHW
    if duplicate: 
        depth_map = np.repeat(depth_map[:, :, np.newaxis], 3, axis=2)
        depth_map = torch.from_numpy(depth_map).permute(2, 0, 1).unsqueeze(dim=0)
    else: 
        # One channel, Convert numpy into BCHW
        depth_map = torch.from_numpy(depth_map).unsqueeze(dim=0).unsqueeze(dim=0)

    # Downsampling, Nearest Interpolation
    resized_depth_map = torch.nn.functional.interpolate(depth_map, size=(resized_img_size, resized_img_size),
                                                        mode='nearest')
    return resized_depth_map

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

def sun3d_get_depth_image_list(data_path):
    xtion_depth_path = os.path.join(data_path, 'xtion/sun3ddata/*/*/*/depth/*.png') 
    other_depth_path = os.path.join(data_path, '*/*/*/depth/*.png')
    depth_list_sun3d = glob.glob(xtion_depth_path)
    depth_list_other = glob.glob(other_depth_path)

    #depth_list_sun3d = glob.glob(r"/disk2/SUNRGBD/xtion/sun3ddata/*/*/*/depth/*.png")
    #depth_list_other = glob.glob(r"/disk2/SUNRGBD/*/*/*/depth/*.png")
    depth_list = depth_list_sun3d + depth_list_other
    return depth_list

def sun3d_get_rgb_image_list(data_path):
    xtion_img_path = os.path.join(data_path, 'xtion/sun3ddata/*/*/*/image/*.jpg')
    image_list_sun3d = glob.glob(r"/disk2/SUNRGBD/xtion/sun3ddata/*/*/*/image/*.jpg")
    image_list_other = glob.glob(r"/disk2/SUNRGBD/*/*/*/image/*.jpg")
    # print(len(image_list_other))
    # print(len(image_list_sun3d))
    image_list = image_list_sun3d + image_list_other
    return image_list

def randAugmenter():
    aug_ind = np.random.choice(np.arange(len(augmenters_DREAM)), 3, replace=False)
    aug = iaa.Sequential([augmenters_DREAM[aug_ind[0]],
                            augmenters_DREAM[aug_ind[1]],
                            augmenters_DREAM[aug_ind[2]]]
                            )
    return aug

def normlize(numpy):
    numpy_norm = (numpy - np.min(numpy)) / (np.max(numpy) - np.min(numpy) + 1e-8)
    return numpy_norm

def augment_image_DREAM(image, depth, anomaly_rgb, anomaly_depth, resize_shape=[256,256]):
    
    aug = randAugmenter()
    perlin_scale = 6
    min_perlin_scale = 0
    anomaly_source_img = cv2.imread(anomaly_rgb)
    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(resize_shape[1], resize_shape[0]))
    anomaly_source_depth = cv2.imread(anomaly_depth)
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

def transform_image_perlin(image_path, depth_path, resize_shape=[256,256]):
    rgb_anomaly_source_list = get_rgb_image_list()
    depth_anomaly_source_list = get_depth_image_list()
    anomaly_source_idx = torch.randint(0, len(rgb_anomaly_source_list), (1,)).item()
    
    image = cv2.imread(image_path)
    image = cv2.resize(image, dsize=(resize_shape[1], resize_shape[0]))
    depth = read_tiff(depth_path)
    depth = cv2.resize(depth, dsize=(resize_shape[1], resize_shape[0]))

    do_aug_orig = torch.rand(1).numpy()[0] > 0.7
    if do_aug_orig:
        image = rot_DREAM(image=image)
        depth = rot_DREAM(image=depth)

    image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32)
    depth = np.array(depth).reshape((depth.shape[0], depth.shape[1], depth.shape[2])).astype(np.float32)
    image = normlize(image)
    depth = normlize(depth)
    # image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
    # depth = np.array(depth).reshape((depth.shape[0], depth.shape[1], depth.shape[2])).astype(np.float32) * 2.0
    augmented_image, augmented_depth, anomaly_mask, has_anomaly = augment_image_DREAM(image, depth, anomaly_rgb=rgb_anomaly_source_list[anomaly_source_idx],
                                                                        anomaly_depth=depth_anomaly_source_list[anomaly_source_idx])
    augmented_image = np.transpose(augmented_image, (2, 0, 1))
    augmented_depth = np.transpose(augmented_depth, (2, 0, 1))
    # image = np.transpose(image, (2, 0, 1))
    anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
    # return image, augmented_image, anomaly_mask, has_anomaly
    # return torch.from_numpy(image), torch.from_numpy(augmented_image), torch.from_numpy(anomaly_mask), torch.from_numpy(has_anomaly)
    return torch.from_numpy(augmented_image), torch.from_numpy(augmented_depth[0,:,:]), torch.from_numpy(anomaly_mask), torch.from_numpy(has_anomaly)

def transform_image_DREAM_noperlin(image, resize_shape=[256,256]):
    if resize_shape != None:
        image = cv2.resize(image, dsize=(resize_shape[1], resize_shape[0]))
    image = normlize(image)
    image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)

def transform_depth_DREAM_noperlin(image, resize_shape=[256,256], depth_duplicate=1):
    if resize_shape != None:
        image = cv2.resize(image, dsize=(resize_shape[1], resize_shape[0]))
    image = normlize(image)
    image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    if(depth_duplicate==1):
        return torch.from_numpy(image[0,:,:])
    else:
        return torch.from_numpy(image)

def aug_DREAM_3D(x, xyz, mask, y, phase='train', depth_duplicate=1, resize_shape=[256,256]):
    if phase=='train':
        x, depth_map, mask, y = transform_image_perlin(x, xyz, resize_shape=resize_shape)
    elif (phase == 'test' and y == 0):
        x = cv2.imread(x)
        x = transform_image_DREAM_noperlin(x)
        mask = torch.zeros([1, x.shape[1], x.shape[2]])
        tiff_img = read_tiff(xyz)
        depth_map = transform_depth_DREAM_noperlin(tiff_img, resize_shape=resize_shape, depth_duplicate=depth_duplicate)
        y = np.array([0], dtype=np.float32)
        y = torch.from_numpy(y)
    else:
        x = cv2.imread(x)
        x = transform_image_DREAM_noperlin(x)
        mask = cv2.imread(mask)
        mask = transform_image_DREAM_noperlin(mask)
        tiff_img = read_tiff(xyz)
        depth_map = transform_depth_DREAM_noperlin(tiff_img, resize_shape=resize_shape, depth_duplicate=depth_duplicate)
        y = np.array([1], dtype=np.float32)
        y = torch.from_numpy(y)
    
    return x, y, mask, depth_map