import torch
import os
import math
import random
from PIL import Image
from torch.utils.data import Dataset


__all__ = ['MVTec2DF3D', 'mvtec2df3d_classes']

def mvtec2df3d_classes():
    return [ "bagel", "cable_gland", "carrot", "cookie", "dowel",
             "foam", "peach", "potato", "rope", "tire"]


class MVTec2DF3D(Dataset):
    def __init__(self, data_path, learning_mode='centralized', phase='train', 
                 data_transform=None, num_task=10):

        self.data_path = data_path
        self.learning_mode = learning_mode
        self.phase = phase
        self.img_transform = data_transform[0]
        self.mask_transform = data_transform[1] 
        self.class_name = mvtec2df3d_classes()
        assert set(self.class_name) <= set(mvtec2df3d_classes())
        
        self.num_task = num_task 
        self.class_in_task = []

        self.imgs_list = []
        self.labels_list = []
        self.masks_list = []
        self.task_ids_list = []
        
        # mark each sample task id
        self.sample_num_in_task = []
        self.sample_indices_in_task = []

        # load dataset
        self.load_dataset()
        self.allocate_task_data()

    def __getitem__(self, idx):
        img_src, label, mask, task_id = self.imgs_list[idx], self.labels_list[idx], self.masks_list[idx], self.task_ids_list[idx]

        img = Image.open(img_src).convert('RGB')
        img = self.img_transform(img)

        if label == 0:
            if isinstance(img, tuple):
                mask = torch.zeros([1, img[0].shape[1], img[0].shape[2]])
            else:
                mask = torch.zeros([1, img.shape[1], img.shape[2]])
        else:
            mask = Image.open(mask)
            mask = self.mask_transform(mask)

        return {
            'img': img, 'label': label, 'mask': mask, 'task_id': task_id, 'img_src': img_src,
        }

    def __len__(self):
        return len(self.imgs_list)

    def load_dataset(self):
        # input x, label y, [0, 1], good is 0 and bad is 1, mask is ground truth
        # train directory: only good cases
        # test directory: bad and good cases 
        # ground truth directory: only bad case

        # get classes in each task group

        # If num_task is 10, each task constain each class
        self.class_in_task = self.split_chunks(self.class_name, self.num_task)
        # get data
        for id, class_in_task in enumerate(self.class_in_task):
            x, y, mask = [], [], []
            for class_name in class_in_task:
                img_dir = os.path.join(self.data_path, class_name, self.phase)
                gt_dir = os.path.join(self.data_path, class_name, 'ground_truth') 

                img_types = sorted(os.listdir(img_dir))
                for img_type in img_types:

                    # load images
                    img_type_dir = os.path.join(img_dir, img_type)
                    if not os.path.isdir(img_type_dir):
                        continue
                    img_path_list = sorted([os.path.join(img_type_dir, f)
                                            for f in os.listdir(img_type_dir)
                                            if f.endswith('.png')])
                    x.extend(img_path_list)

                    if img_type == 'good':
                        y.extend([0] * len(img_path_list))
                        mask.extend([None] * len(img_path_list))
                    else:
                        y.extend([1] * len(img_path_list))
                        gt_type_dir = os.path.join(gt_dir, img_type)
                        img_name_list = [os.path.splitext(os.path.basename(f))[0] for f in img_path_list]
                        gt_path_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                        for img_fname in img_name_list]
                        mask.extend(gt_path_list)
            
            task_id = [id for i in range(len(x))]
            self.sample_num_in_task.append(len(x))

            self.imgs_list.extend(x)
            self.labels_list.extend(y)
            self.masks_list.extend(mask)
            self.task_ids_list.extend(task_id) 

    def allocate_task_data(self):
        start = 0
        for num in self.sample_num_in_task:
            end = start + num
            indice = [i for i in range(start, end)]
            random.shuffle(indice)
            self.sample_indices_in_task.append(indice)
            start = end

    # split the arr into n chunks
    @staticmethod
    def split_chunks(arr, m):
        n = int(math.ceil(len(arr) / float(m)))
        return [arr[i:i + n] for i in range(0, len(arr), n)]