import numpy as np
import os
import tifffile
import torch
from os.path import join
from tqdm import tqdm
import config as c
from model import FeatureExtractor
from utils import *


def get_neighbor_mean(img, p):
    n_neighbors = np.sum(img[p[0] - 1: p[0] + 2, p[1] - 1: p[1] + 2] > 0)
    if n_neighbors == 0:
        return None
    nb_mean = np.sum(img[p[0] - 1: p[0] + 2, p[1] - 1: p[1] + 2], axis=(0, 1)) / n_neighbors
    return nb_mean


def fill_gaps(img):
    new_img = np.copy(img)
    zero_pixels = np.where(img == 0)
    for x, y in zip(*zero_pixels):
        if img[x, y] == 0:
            nb_mean = get_neighbor_mean(img, [x, y])
            if nb_mean is not None:
                new_img[x, y] = nb_mean
    return new_img


def get_corner_points(img):
    upper_left = np.sum(img[:2, :2]) / np.sum(img[:2, :2] > 0)
    upper_right = np.sum(img[-2:, :2]) / np.sum(img[-2:, :2] > 0)
    lower_left = np.sum(img[:2, -2:]) / np.sum(img[:2, -2:] > 0)
    lower_right = np.sum(img[-2:, -2:]) / np.sum(img[-2:, -2:] > 0)
    return upper_left, upper_right, lower_left, lower_right


def remove_background(img, bg_thresh):
    w, h = img.shape[:2]
    upper_left, upper_right, lower_left, lower_right = get_corner_points(img)
    x_top = np.linspace(upper_left, upper_right, w)
    x_bottom = np.linspace(lower_left, lower_right, w)
    top_ratio = np.linspace(1, 0, h)[None]
    bottom_ratio = np.linspace(0, 1, h)[None]
    background = x_top[:, None] * top_ratio + x_bottom[:, None] * bottom_ratio
    foreground = np.zeros_like(img)
    foreground[np.abs(background - img) > bg_thresh] = 1
    foreground[img == 0] = 0
    return foreground


def preprocess_3D(base_dir, n_fills, bg_thresh):
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(join(base_dir, d))]
    for c in classes:
        print(c)
        class_dir = join(base_dir, c)
        for set in ["train", "test"]:
            print('\t' + set)
            set_dir = join(class_dir, set)
            subclass = os.listdir(set_dir)
            for sc in subclass:
                print('\t\t' + sc)
                sub_dir = join(set_dir, sc, 'xyz')
                samples = os.listdir(sub_dir)
                save_dir = join(set_dir, sc, 'z')
                os.makedirs(save_dir, exist_ok=True)
                for i_s, s in enumerate(tqdm(samples)):
                    s_path = join(sub_dir, s)
                    img = tifffile.imread(s_path)
                    img = img[:, :, -1] # get z component
                    for _ in range(n_fills):
                        img = fill_gaps(img)
                    mask = remove_background(img, bg_thresh)
                    sample = np.stack([img, mask], axis=2)
                    np.save(join(save_dir, s[:s.find('.')]), sample)


def extract_image_features(base_dir, extract_layer=c.extract_layer):
    model = FeatureExtractor(layer_idx=extract_layer)
    model.to(c.device)
    model.eval()

    
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(join(base_dir, d))]

    for class_name in classes:
        print(class_name)
        train_set, test_set = load_img_datasets(base_dir, class_name)
        train_loader, test_loader = make_dataloaders(train_set, test_set, shuffle_train=False, drop_last=False)
        for name, loader in zip(['train', 'test'], [train_loader, test_loader]):
            features = list()
            for i, data in enumerate(tqdm(loader)):
                img = data[0].to(c.device)
                with torch.no_grad():
                    z = model(img)
                features.append(t2np(z))

            features = np.concatenate(features, axis=0)
            export_dir = join(c.feature_dir, class_name)
            os.makedirs(export_dir, exist_ok=True)
            print(export_dir)
            np.save(join(export_dir, f'{name}.npy'), features)


extract_image_features(c.dataset_dir)
if c.use_3D_dataset:
    preprocess_3D(c.dataset_dir, n_fills=c.n_fills, bg_thresh=c.bg_thresh)
