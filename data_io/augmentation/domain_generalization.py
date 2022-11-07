import torch
from PIL import Image
from torchvision import transforms as T
import kornia.geometry.transform as kt


def domain_gen(config, data):
    data_dg = []
    degrees = [(0, 0), (90, 90), (180, 180), (270, 270)]
    # degrees = [(0, 0), (45, 45), (90, 90), (135, 135), (180, 180), (225, 225), (270, 270), (315, 315)]
    for d in data:
        img_src = d['img_src']
        img = Image.open(img_src).convert('RGB')
        mask = d['mask']
        for degree in degrees:
            t = {'degree': degree, 'translate': [0, 0], 'scale': [1.0, 1.0],
                'size': [config['data_size'], config['data_size']], 'crop_size': [config['data_crop_size'], config['data_crop_size']]}
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

def feature_augmentation(features, device):
    assert len(features) > 0, 'Feature Augmentation should be done in Original Features'
    #angles_list = [45, 90, 135, 180, 225, 270, 315, 360]
    angles_list = [45.0, 135.0, 225.0]

    rot_feat_1 = features[0]
    rot_feat_2 = features[1]

    for angle in angles_list:
        angle = torch.tensor(angle).to(device)
        rot_feat_1 = torch.cat((rot_feat_1, kt.rotate(features[0], angle)), dim=0)
        rot_feat_2 = torch.cat((rot_feat_2, kt.rotate(features[1], angle)), dim=0)
        
    feature_rot = [rot_feat_1, rot_feat_2]

    return feature_rot