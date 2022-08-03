import torch
import os
import random
from PIL import Image
import numpy as np
from torchvision import models
from torchvision import transforms as T
from data_io.mvtec2d import FewShot
from torch.utils.data import DataLoader
from arch_base.patchcore2d import PatchCore2D
import cv2
import torchvision



def domain_gen(config, data):
    data_dg = []
    degrees = [(0, 0), (90, 90), (180, 180), (270, 270)]
    # degrees = [(0, 0), (45, 45), (90, 90), (135, 135), (180, 180), (225, 225), (270, 270), (315, 315)]
    for d in data:
        img_src = d['img_src']
        img = Image.open(img_src).convert('RGB')
        mask = d['mask']
        for degree in degrees:
            t = {'degree': degree, 'translate': [0, 0], 'scale': [1.0, 1.0], 'size': [256, 256], 'crop_size': [224, 224]}
            imge_transform = T.Compose([T.Resize(t['size']),
                                        T.CenterCrop(t['crop_size']),
                                        T.RandomAffine(degrees=t['degree']),
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            mask_transform = T.Compose([T.ToPILImage(),
                                        T.RandomAffine(degrees=t['degree']),
                                        T.ToTensor()])
            img_da = imge_transform(img)
            mask = mask_transform(mask)
            data_dg.append({'img': img_da, 'label': d['label'], 'mask': mask, 'task_id': d['task_id'], 'img_src': img_src})

    return data_dg            

