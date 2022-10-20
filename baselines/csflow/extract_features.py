import numpy as np
import torch
from tqdm import tqdm
import config as c
from model import FeatureExtractor
from utils import *
import os


def extract(train_loader, test_loader, class_name):
    model = FeatureExtractor()
    model.to(c.device)
    model.eval()
    with torch.no_grad():
        for name, loader in zip(['train', 'test'], [train_loader, test_loader]):
            features = [list() for _ in range(c.n_scales)]
            labels = list()
            for i, data in enumerate(tqdm(loader)):
                inputs, l = preprocess_batch(data)
                labels.append(t2np(l))
                z = model(inputs)
                for iz, zi in enumerate(z):
                    features[iz].append(t2np(zi))

            for i_f, f in enumerate(features):
                f = np.concatenate(f, axis=0)
                np.save(export_dir + class_name + '_scale_' + str(i_f) + '_' + name, f)
            if name == 'test':
                labels = np.concatenate(labels)
                np.save(export_dir + class_name + '_labels', labels)


export_name = c.class_name
export_dir = 'data/features/' + export_name + '/'
c.pre_extracted = False
os.makedirs(export_dir, exist_ok=True)
train_set, test_set = load_datasets(c.dataset_path, c.class_name)
train_loader, test_loader = make_dataloaders(train_set, test_set)
extract(train_loader, test_loader, c.class_name)
paths = [p for p, l in test_set.samples]
np.save(export_dir + c.class_name + '_image_paths.npy', paths)