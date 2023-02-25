import torch
from PIL import Image
from torchvision import transforms as T
import kornia.geometry.transform as kt
import numpy as np


def domain_gen(config, data):
    data_dg = []
    aug_num = config['fewshot_num_dg']
    size = config['data_size']
    crop_size = config['data_crop_size']

    if config['fewshot_aug_type'] == 'rotation':
        degrees = np.linspace(0, 360, aug_num + 1, endpoint=False)
        for d in data:
            img_src = d['img_src']
            img = Image.open(img_src).convert('RGB')
            mask = d['mask']
            for degree in degrees:
                img_transform = T.Compose([T.Resize((size, size)),
                                           T.CenterCrop(crop_size),
                                           T.RandomAffine((degree, degree)),
                                           T.ToTensor(),
                                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                mask_transform = T.Compose([T.ToPILImage(),
                                            T.RandomAffine((degree, degree)),
                                            T.ToTensor()])
                img_da = img_transform(img)
                mask = mask_transform(mask)
                data_dg.append(
                    {'img': img_da, 'label': d['label'], 'mask': mask, 'task_id': d['task_id'], 'img_src': img_src})

    elif config['fewshot_aug_type'] == 'scale':
        scales = np.linspace(0.5, 1, aug_num + 1)
        for d in data:
            img_src = d['img_src']
            img = Image.open(img_src).convert('RGB')
            mask = d['mask']
            for scale in scales:
                img_transform = T.Compose([T.Resize((size, size)),
                                           T.CenterCrop(crop_size),
                                           T.RandomAffine((0, 0), scale=(scale, scale)),
                                           T.ToTensor(),
                                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                mask_transform = T.Compose([T.ToPILImage(),
                                            T.RandomAffine((0, 0), scale=(scale, scale)),
                                            T.ToTensor()])
                img_da = img_transform(img)
                mask = mask_transform(mask)
                data_dg.append(
                    {'img': img_da, 'label': d['label'], 'mask': mask, 'task_id': d['task_id'], 'img_src': img_src})

    elif config['fewshot_aug_type'] == 'translate':
        translates = np.linspace(0, 0.5, aug_num + 1)
        for d in data:
            img_src = d['img_src']
            img = Image.open(img_src).convert('RGB')
            mask = d['mask']
            for translate in translates:
                img_transform = T.Compose([T.Resize((size, size)),
                                           T.CenterCrop(crop_size),
                                           T.RandomAffine((0, 0), translate=(translate, translate)),
                                           T.ToTensor(),
                                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                mask_transform = T.Compose([T.ToPILImage(),
                                            T.RandomAffine((0, 0), translate=(translate, translate)),
                                            T.ToTensor()])
                img_da = img_transform(img)
                mask = mask_transform(mask)
                data_dg.append(
                    {'img': img_da, 'label': d['label'], 'mask': mask, 'task_id': d['task_id'], 'img_src': img_src})

    elif config['fewshot_aug_type'] == 'flip':
        assert aug_num == 4, 'Only 4 flips are supported'
        flips = [[T.RandomHorizontalFlip(p=1)], [T.RandomVerticalFlip(p=1)],
                 [T.RandomHorizontalFlip(p=1), T.RandomVerticalFlip(p=1)],
                 [T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)]]
        for d in data:
            img_src = d['img_src']
            img = Image.open(img_src).convert('RGB')
            mask = d['mask']
            for flip in flips:
                img_transform = T.Compose([T.Resize((size, size)),
                                           T.CenterCrop(crop_size),
                                           *flip,
                                           T.ToTensor(),
                                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                mask_transform = T.Compose([T.ToPILImage(),
                                            *flip,
                                            T.ToTensor()])
                img_da = img_transform(img)
                mask = mask_transform(mask)
                data_dg.append(
                    {'img': img_da, 'label': d['label'], 'mask': mask, 'task_id': d['task_id'], 'img_src': img_src})

    elif config['fewshot_aug_type'] == 'color_jitter':
        brightness = np.linspace(0.5, 1.5, aug_num)
        brightness = np.concatenate((brightness, np.array([1])))
        contrast = np.linspace(0.5, 1.5, aug_num)
        contrast = np.concatenate((contrast, np.array([1])))
        for d in data:
            img_src = d['img_src']
            img = Image.open(img_src).convert('RGB')
            mask = d['mask']
            for i in range(aug_num + 1):
                img_transform = T.Compose([T.Resize((size, size)),
                                           T.CenterCrop(crop_size),
                                           T.ColorJitter(brightness=brightness[i], contrast=contrast[i]),
                                           T.ToTensor(),
                                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                mask_transform = T.Compose([T.ToPILImage(),
                                            T.ColorJitter(brightness=brightness[i], contrast=contrast[i]),
                                            T.ToTensor()])
                img_da = img_transform(img)
                mask = mask_transform(mask)
                data_dg.append(
                    {'img': img_da, 'label': d['label'], 'mask': mask, 'task_id': d['task_id'], 'img_src': img_src})

    elif config['fewshot_aug_type'] == 'perspective':
        perspectives = np.linspace(0, 0.5, aug_num + 1)
        for d in data:
            img_src = d['img_src']
            img = Image.open(img_src).convert('RGB')
            mask = d['mask']
            for perspective in perspectives:
                img_transform = T.Compose([T.Resize((size, size)),
                                           T.CenterCrop(crop_size),
                                           T.RandomPerspective(distortion_scale=perspective),
                                           T.ToTensor(),
                                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                mask_transform = T.Compose([T.ToPILImage(),
                                            T.RandomPerspective(distortion_scale=perspective),
                                            T.ToTensor()])
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
