import torch
import os

data_path = '/disk/mvtec/2D'
class_name = 'bottle'
#phase = 'train'
phase = 'test'

img_dir = os.path.join(data_path, class_name, phase)
gt_dir = os.path.join(data_path, class_name, 'ground_truth')

img_types = sorted(os.listdir(img_dir))
print(img_types)

img_type = 'good'

img_type_dir = os.path.join(img_dir, img_type)

img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])

print(img_fpath_list)