import torch
from PIL import Image
from torchvision import transforms as T
import kornia.geometry.transform as kt
import numpy as np
from augmentation.cutpaste_aug import *


def domain_gen(config, data):
    data_dg = []
    aug_num = config['fewshot_num_dg']
    size = config['data_size']
    crop_size = config['data_crop_size']
    after_cutpaste_transform = T.Compose([T.RandomRotation(90),
                                          T.ToTensor(),
                                          T.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                                          ])
    # assert at least one fewshot_aug_type is selected
    assert 'rotation' in config['fewshot_aug_type'] or 'scale' in config['fewshot_aug_type'] or \
           'translate' in config['fewshot_aug_type'] or 'flip' in config['fewshot_aug_type'] or \
           'color_jitter' in config['fewshot_aug_type'] or 'perspective' in config['fewshot_aug_type'], \
        'At least one fewshot_aug_type should be selected'
    if config['train_aug_type'] == 'cutpaste':
        img_transform_list = [T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                              T.Resize((crop_size, crop_size)),
                              CutPasteNormal(transform=after_cutpaste_transform)]
    else:
        img_transform_list = [T.Resize((size, size)),
                              T.CenterCrop(crop_size),
                              T.ToTensor(),
                              T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    mask_transform_list = [T.ToPILImage(), T.ToTensor()]
    degrees = np.linspace(0, 360, aug_num + 1, endpoint=False)
    scales = np.linspace(1, 0.5, aug_num + 1)
    translates = np.linspace(0, 0.5, aug_num + 1)
    flips = [[T.RandomHorizontalFlip(p=1)], [T.RandomVerticalFlip(p=1)],
             [T.RandomHorizontalFlip(p=1), T.RandomVerticalFlip(p=1)],
             [T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)]]
    brightness = np.linspace(0.5, 1.5, aug_num)
    brightness = np.concatenate((np.array([1]), brightness))
    contrast = np.linspace(0.5, 1.5, aug_num)
    contrast = np.concatenate((np.array([1]), contrast))
    perspectives = np.linspace(0, 0.5, aug_num + 1)
    for d in data:
        img_src = d['img_src']
        img = Image.open(img_src).convert('RGB')
        mask = d['mask']
        for i in range(aug_num + 1):
            img_transforms = img_transform_list
            mask_transforms = mask_transform_list
            if 'rotation' in config['fewshot_aug_type']:
                img_transforms.insert(2, T.RandomAffine((degrees[i], degrees[i])))
                mask_transforms.insert(1, T.RandomAffine((degrees[i], degrees[i])))
            if 'scale' in config['fewshot_aug_type']:
                img_transforms.insert(2, T.RandomAffine((0, 0), scale=(scales[i], scales[i])))
                mask_transforms.insert(1, T.RandomAffine((0, 0), scale=(scales[i], scales[i])))
            if 'translate' in config['fewshot_aug_type']:
                img_transforms.insert(2,
                                      T.RandomAffine((0, 0), translate=(translates[i], translates[i])))
                mask_transforms.insert(1, T.RandomAffine((0, 0),
                                                         translate=(translates[i], translates[i])))
            if 'flip' in config['fewshot_aug_type']:
                if i > 0:
                    for flip in flips[min(i-1, 3)]:
                        img_transforms.insert(2, flip)
                        mask_transforms.insert(1, flip)
            if 'color_jitter' in config['fewshot_aug_type']:
                img_transforms.insert(2, T.ColorJitter(brightness=brightness[i], contrast=contrast[i]))
                mask_transforms.insert(1,
                                       T.ColorJitter(brightness=brightness[i], contrast=contrast[i]))
            if 'perspective' in config['fewshot_aug_type']:
                img_transforms.insert(2, T.RandomPerspective(distortion_scale=perspectives[i], p=1))
                mask_transforms.insert(1, T.RandomPerspective(distortion_scale=perspectives[i], p=1))
            img_transform = T.Compose(img_transforms)
            mask_transform = T.Compose(mask_transforms)
            img_da = img_transform(img)
            mask = mask_transform(mask)
            data_dg.append(
                {'img': img_da, 'label': d['label'], 'mask': mask, 'task_id': d['task_id'], 'img_src': img_src})
    return data_dg


def feature_augmentation(features, device):
    assert len(features) > 0, 'Feature Augmentation should be done in Original Features'
    # angles_list = [45, 90, 135, 180, 225, 270, 315, 360]
    angles_list = [45.0, 135.0, 225.0]

    rot_feat_1 = features[0]
    rot_feat_2 = features[1]

    for angle in angles_list:
        angle = torch.tensor(angle).to(device)
        rot_feat_1 = torch.cat((rot_feat_1, kt.rotate(features[0], angle)), dim=0)
        rot_feat_2 = torch.cat((rot_feat_2, kt.rotate(features[1], angle)), dim=0)

    feature_rot = [rot_feat_1, rot_feat_2]

    return feature_rot
