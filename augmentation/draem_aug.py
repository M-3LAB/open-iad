import torch
import numpy as np
import glob
import cv2
import imgaug.augmenters as iaa
from augmentation.perlin import rand_perlin_2d_np


__all__ = ['DraemAugData']

class DraemAugData():
    def __init__(self, anomaly_source_path='/dtd/images', resize_shape=[256, 256]):
        self.resize_shape = resize_shape

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
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
        self.aug = self.randAugmenter()
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + '/*/*.jpg')) 

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def transform_batch(self, images, labels, masks):
        augmented_images, anomaly_masks, has_anomalys = [], [], []
        for i in range(len(images)):
            image = images.numpy()[i].transpose(1, 2, 0) 
            label = labels.numpy()[i]
            mask = masks.numpy()[i] 
            augmented_image, anomaly_mask, has_anomaly = self.transform_img(image, label, mask)
            augmented_images.append(augmented_image)
            anomaly_masks.append(anomaly_mask)
            has_anomalys.append(has_anomaly)

        np.concatenate(augmented_images, axis=0)
        np.concatenate(anomaly_masks, axis=0)
        np.concatenate(has_anomalys, axis=0)

        augmented_images = torch.from_numpy(np.array(augmented_images))
        anomaly_masks = torch.from_numpy(np.array(anomaly_masks))
        has_anomalys = torch.from_numpy(np.array(has_anomalys))

        return augmented_images, anomaly_masks, has_anomalys

    def transform_img(self, image, label, mask):
        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0

        
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, self.anomaly_source_paths[anomaly_source_idx])
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))

        return augmented_image, anomaly_mask, has_anomaly 

    def augment_image(self, image, anomaly_source_path):
        perlin_scale = 6
        min_perlin_scale = 0

        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = self.aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8
        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            mask = (perlin_thr).astype(np.float32)
            augmented_image = mask * augmented_image + (1 - mask) * image
            has_anomaly = 1.0
            if np.sum(mask) == 0:
                has_anomaly = 0.0
            return augmented_image, mask, np.array([has_anomaly], dtype=np.float32)