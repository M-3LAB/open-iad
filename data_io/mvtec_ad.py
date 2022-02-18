import torch
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_io.augmentation import mvtec_2d_resize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

__all__ = ['MVTec2D', 'MVTec3D', 'mvtec_2d_classes', 'mvtec_3d_classes', 'MVTec2DContinualList', 'MVTec2DContinualDataloaderList']

def mvtec_2d_classes():
    return [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

def mvtec_3d_classes():
    return [
        "bagel",
        "cable_grand",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]

def MVTec2DContinualList(data_path, class_name, num_continual_tasks, phase, mode, data_transform=None, mask_transform=None):
    MVTecContinualList = []
    className = mvtec_2d_classes()
    for i in range(0, int(15/num_continual_tasks)):
        sub_class_name = className[num_continual_tasks*i: num_continual_tasks*i+num_continual_tasks]
        MVTecContinualList.append(MVTec2DContinual(data_path, sub_class_name, i, mode, phase, 
                 data_transform=None, mask_transform=None))
    if 15%num_continual_tasks != 0:
        sub_class_name = className[int(15/num_continual_tasks)*num_continual_tasks: 15]
        MVTecContinualList.append(MVTec2DContinual(data_path, sub_class_name, int(15/num_continual_tasks), mode, phase, 
                    data_transform=None, mask_transform=None))

    return MVTecContinualList

def MVTec2DContinualDataloaderList(MVTecContinualList, batch_size, shuffle, num_workers):
    MVTecContinualDataloaderList = []
    for mvtecDataset in MVTecContinualList:
        train_loader = DataLoader(mvtecDataset, batch_size, shuffle, num_workers)
        MVTecContinualDataloaderList.append(train_loader)
    
    return MVTecContinualDataloaderList

class MVTec2D(Dataset):
    def __init__(self, data_path, class_name, mode='centralized', phase='train', 
                 data_transform=None, mask_transform=None):

        self.data_path = data_path
        self.mode = mode
        self.phase = phase
        self.data_transform = data_transform 
        self.mask_transform = mask_transform
        self.class_name = class_name
        assert set(self.class_name) <= set(mvtec_2d_classes())
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # data preprocessing 
        self.data_transform = data_transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = mvtec_2d_resize(x)

        if y == 0:
            mask = torch.zeros([1, x.shape[1], x.shape[2]])
        else:
            mask = Image.open(mask)
            mask = self.mask_transforms(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        # input x, label y, [0, 1], good is 0 and bad is 1, mask is ground truth
        x, y, mask = [], [], []
        for sub_class_name in self.class_name:
            img_dir = os.path.join(self.data_path, sub_class_name, self.phase)
            gt_dir = os.path.join(self.data_path, sub_class_name, 'ground_truth') 

            img_types = sorted(os.listdir(img_dir))
            for img_type in img_types:

                # load images
                img_type_dir = os.path.join(img_dir, img_type)
                if not os.path.isdir(img_type_dir):
                    continue
                img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                        for f in os.listdir(img_type_dir)
                                        if f.endswith('.png')])
                x.extend(img_fpath_list)

                """
                Train Directory: Only Good Cases
                Test Directory: Bad and Good Cases 
                Ground Truth Directory: Only Bad Case
                Detail Can be referred from MVTec 2D dataset
                """
                if img_type == 'good':
                    y.extend([0] * len(img_fpath_list))
                    mask.extend([None] * len(img_fpath_list))
                else:
                    y.extend([1] * len(img_fpath_list))
                    gt_type_dir = os.path.join(gt_dir, img_type)
                    img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                    for img_fname in img_fname_list]
                    mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'
        return list(x), list(y), list(mask)

class MVTec2DContinual(Dataset):#taskId is from 0 to floor(15/num_tasks_continual)
    def __init__(self, data_path, class_name, taskId, mode='continual', phase='train', 
                 data_transform=None, mask_transform=None):

        self.data_path = data_path
        self.mode = mode
        self.phase = phase
        self.data_transform = data_transform 
        self.mask_transform = mask_transform
        self.class_name = class_name
        assert set(self.class_name) <= set(mvtec_2d_classes())
        # load dataset
        self.x, self.y, self.mask, self.taskId = self.load_dataset_folder(taskId)

        # data preprocessing 
        self.data_transform = data_transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx):
        x, y, mask, taskId = self.x[idx], self.y[idx], self.mask[idx], self.taskId[idx]

        x = Image.open(x).convert('RGB')
        x = mvtec_2d_resize(x)

        if y == 0:
            mask = torch.zeros([x.size[0], x.size[1]])
        else:
            mask = Image.open(mask)
            # mask = self.mask_transforms(mask)

        x = np.array(x)
        mask = np.array(mask)

        # check results
        plt.subplot(121)
        plt.imshow(x) 
        plt.title('input')
        plt.subplot(122)
        plt.title('mask')
        plt.imshow(mask) 
        plt.savefig('./legacy_code/img_{}.jpg'.format('result'))
        plt.close()

        return x, y, mask, taskId

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self, taskId):
        # input x, label y, [0, 1], good is 0 and bad is 1, mask is ground truth
        x, y, mask = [], [], []
        for sub_class_name in self.class_name:
            img_dir = os.path.join(self.data_path, sub_class_name, self.phase)
            gt_dir = os.path.join(self.data_path, sub_class_name, 'ground_truth') 

            img_types = sorted(os.listdir(img_dir))
            for img_type in img_types:

                # load images
                img_type_dir = os.path.join(img_dir, img_type)
                if not os.path.isdir(img_type_dir):
                    continue
                img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                        for f in os.listdir(img_type_dir)
                                        if f.endswith('.png')])
                x.extend(img_fpath_list)

                """
                Train Directory: Only Good Cases
                Test Directory: Bad and Good Cases 
                Ground Truth Directory: Only Bad Case
                Detail Can be referred from MVTec 2D dataset
                """
                if img_type == 'good':
                    y.extend([0] * len(img_fpath_list))
                    mask.extend([None] * len(img_fpath_list))
                else:
                    y.extend([1] * len(img_fpath_list))
                    gt_type_dir = os.path.join(gt_dir, img_type)
                    img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                    for img_fname in img_fname_list]
                    mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'
        tasksId = len(x)*[taskId] 
        return list(x), list(y), list(mask), list(tasksId)


class MVTec3D(Dataset):
    def __init__(self, data_path, class_name):
        self.data_path = data_path
        self.class_name = class_name

    def __getitem__(self):
        pass