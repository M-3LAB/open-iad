import torch
import os
import math
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


__all__ = ['MVTec3D', 'mvtec3d_classes']


def mvtec3d_classes():
    return [ "bagel", "cable_gland", "carrot", "cookie", "dowel",
             "foam", "peach", "potato", "rope", "tire"]


class MVTec3D(Dataset):
    def __init__(self, data_path, learning_mode='centralized', phase='train', 
                 data_transform=None, num_task=5):

        self.data_path = data_path
        self.learning_mode = learning_mode
        self.phase = phase
        self.data_transform = data_transform
        self.class_name = mvtec3d_classes()
        assert set(self.class_name) <= set(mvtec3d_classes())
        
        self.num_task = num_task 
        self.class_in_task = []

        self.all_x = []
        self.all_y = []
        self.all_mask = []
        self.all_task_id = []
        self.all_xyz = []

        # continual
        self.conti_len = []
        self.continual_indices = []

        # load dataset
        self.load_dataset()
        self.allocate_task_data()

        # data preprocessing 
        self.imge_transform = T.Compose([T.Resize(self.data_transform['data_size']),
                                        T.CenterCrop(self.data_transform['data_size']),
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                                        ])
        self.mask_transform = T.Compose([T.Resize(self.data_transform['mask_size']),
                                        T.CenterCrop(self.data_transform['mask_size']),
                                        T.ToTensor()
                                        ])
    def __getitem__(self, idx):
        x, y, mask, task_id, xyz = self.all_x[idx], self.all_y[idx], self.all_mask[idx], self.all_task_id[idx], self.all_xyz[idx]

        x = Image.open(x).convert('RGB')
        x = self.imge_transform(x)

        if y == 0:
            mask = torch.zeros([1, x.shape[1], x.shape[2]])
        else:
            mask = Image.open(mask)
            mask = self.mask_transform(mask)

        return x, y, mask, task_id, xyz

    def __len__(self):
        return len(self.all_x)


    def load_dataset(self):
        # input x, label y, [0, 1], good is 0 and bad is 1, mask is ground truth
        # train directory: only good cases
        # test directory: bad and good cases 
        # ground truth directory: only bad case

        # get classes in each task group
        self.class_in_task = self.split_chunks(self.class_name, self.num_task)
        # get data
        for id, class_in_task in enumerate(self.class_in_task):
            x, y, mask, xyz  = [], [], [], []
            for class_name in class_in_task:
                img_dir = os.path.join(self.data_path, class_name, self.phase)

                img_types = sorted(os.listdir(img_dir))
                for img_type in img_types:

                    # load images
                    img_type_dir = os.path.join(img_dir, img_type)
                    if not os.path.isdir(img_type_dir):
                        continue
                    img_path_list = sorted([os.path.join(img_type_dir, 'rgb', f)
                                            for f in os.listdir(img_type_dir + '/rgb')
                                            if f.endswith('.png')])
                    x.extend(img_path_list)
                    xyz_path_list = sorted([os.path.join(img_type_dir, 'xyz', f)
                                            for f in os.listdir(img_type_dir + '/xyz')
                                            if f.endswith('.tiff')])


                    if img_type == 'good':
                        y.extend([0] * len(img_path_list))
                        mask.extend([None] * len(img_path_list))
                        # load xyz data
                        xyz.extend(xyz_path_list)
                    else:
                        y.extend([1] * len(img_path_list))
                        gt_type_dir = os.path.join(img_dir, img_type, 'gt')
                        img_name_list = [os.path.splitext(os.path.basename(f))[0] for f in img_path_list]
                        gt_path_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                        for img_fname in img_name_list]
                        mask.extend(gt_path_list)
                        # load xyz data
                        xyz_type_dir = os.path.join(img_dir, img_type, 'xyz')
                        img_name_list = [os.path.splitext(os.path.basename(f))[0] for f in img_path_list]
                        xyz_path_list = [os.path.join(xyz_type_dir, img_fname + '.tiff')
                                        for img_fname in img_name_list]
                        xyz.extend(xyz_path_list)
                        
            # continual
            task_id = [id for i in range(len(x))]
            self.conti_len.append(len(x))

            self.all_x.extend(x)
            self.all_y.extend(y)
            self.all_mask.extend(mask)
            self.all_task_id.extend(task_id) 
            self.all_xyz.extend(xyz) 

    def allocate_task_data(self):
        start = 0
        for num in self.conti_len:
            end = start + num
            indice = [i for i in range(start, end)]
            random.shuffle(indice)
            self.continual_indices.append(indice)
            start = end

    # split the arr into n chunks
    @staticmethod
    def split_chunks(arr, m):
        n = int(math.ceil(len(arr) / float(m)))
        return [arr[i:i + n] for i in range(0, len(arr), n)]