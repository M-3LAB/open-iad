import torch
import os
import math
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import tifffile
import numpy as np
import cv2
import glob
import imgaug.augmenters as iaa
from data_io.augmentation.perlin import rand_perlin_2d_np

__all__ = ['MVTec3D', 'mvtec3d_classes', 'MVTecCL3D', 'read_tiff', 'tiff_to_depth']


def mvtec3d_classes():
    return [ "bagel", "cable_gland", "carrot", "cookie", "dowel",
             "foam", "peach", "potato", "rope", "tire"]

def read_tiff(tiff):
    # tiff_img: numpy format
    tiff_img = tifffile.imread(tiff)
    return tiff_img

def tiff_to_depth(tiff, resized_img_size=224, duplicate=False):
    depth_map = np.array(tiff[:, :, 2])
    # Duplicate depth_map into 3 channels, Convert numpy format into BCHW
    if duplicate: 
        depth_map = np.repeat(depth_map[:, :, np.newaxis], 3, axis=2)
        depth_map = torch.from_numpy(depth_map).permute(2, 0, 1).unsqueeze(dim=0)
    else: 
        # One channel, Convert numpy into BCHW
        depth_map = torch.from_numpy(depth_map).unsqueeze(dim=0).unsqueeze(dim=0)

    # Downsampling, Nearest Interpolation
    resized_depth_map = torch.nn.functional.interpolate(depth_map, size=(resized_img_size, resized_img_size),
                                                        mode='nearest')
    return resized_depth_map

def getFileList(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path,file)
        if os.path.isdir(file_path):
            getFileList(file_path, list_name)
        else:
            list_name.append(file_path)

def get_depth_image_list():
    depth_list_sun3d = glob.glob(r"/disk2/SUNRGBD/xtion/sun3ddata/*/*/*/depth/*.png")
    depth_list_other = glob.glob(r"/disk2/SUNRGBD/*/*/*/depth/*.png")
    # print(len(depth_list_other))
    # print(len(depth_list_sun3d))
    depth_list = depth_list_sun3d + depth_list_other
    return depth_list

def get_rgb_image_list():
    image_list_sun3d = glob.glob(r"/disk2/SUNRGBD/xtion/sun3ddata/*/*/*/image/*.jpg")
    image_list_other = glob.glob(r"/disk2/SUNRGBD/*/*/*/image/*.jpg")
    # print(len(image_list_other))
    # print(len(image_list_sun3d))
    image_list = image_list_sun3d + image_list_other
    return image_list

class MVTec3D(Dataset):
    def __init__(self, data_path, class_names, phase='train', depth_duplicate=False, data_transform=None,
                 perlin=False):
        self.data_path = data_path
        self.phase = phase
        if not isinstance(class_names, list):
            self.class_names = [class_names] 

        self.data_transform = data_transform
        self.depth_duplicate = depth_duplicate
        # no perlin in test dataloader
        if phase=='test' :
            self.perlin=False
        else:
            self.perlin = perlin
        if data_transform!=None:
            self.resize_shape = self.data_transform['data_size']
        else:
            self.resize_shape = [256,256]

        if(perlin==True):
            self.anomaly_image_rgb_list = get_rgb_image_list()
            self.anomaly_image_depth_list = get_depth_image_list()
        
        
        assert set(self.class_names) <= set(mvtec3d_classes()), 'Class is Out of Range'

        """
        x: RGB image
        y: Label, 0: good, 1: bad(anomaly)
        mask: anomaly mask 
        xyz: TIFF image
        """ 

        self.x = []
        self.y = []
        self.mask = []
        self.xyz = []

        # load dataset
        self.load_dataset()

        # data preprocessing 
        self.imge_transform = T.Compose([T.Resize(self.data_transform['data_size'], interpolation=T.InterpolationMode.BICUBIC),
                                        T.CenterCrop(self.data_transform['data_size']),
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                                        ])
        self.mask_transform = T.Compose([T.Resize(self.data_transform['mask_size'], interpolation=T.InterpolationMode.NEAREST),
                                        T.CenterCrop(self.data_transform['mask_size']),
                                        T.ToTensor()
                                        ])
        
        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, depth, anomaly_rgb, anomaly_depth):
        
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_rgb)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))
        anomaly_source_depth = read_tiff(anomaly_depth)
        anomaly_source_depth = cv2.resize(anomaly_source_depth, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        anomaly_depth_augmented = aug(image=anomaly_source_depth)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
        depth_thr = anomaly_depth_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)
        augmented_depth = depth * (1 - perlin_thr) + (1 - beta) * depth_thr + beta * depth * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, depth, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            augmented_depth = augmented_depth.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            augmented_depth = msk * augmented_depth + (1-msk)*depth
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, augmented_depth, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image_perlin(self, image_path, depth_path, rgb_anomaly_source_list = None, depth_anomaly_source_list = None):
        if rgb_anomaly_source_list == None:
            rgb_anomaly_source_list = self.anomaly_image_rgb_list
            depth_anomaly_source_list = self.anomaly_image_depth_list
        anomaly_source_idx = torch.randint(0, len(rgb_anomaly_source_list), (1,)).item()
        
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        depth = read_tiff(depth_path)
        depth = cv2.resize(depth, dsize=(self.resize_shape[1], self.resize_shape[0]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)
            depth = self.rot(image=depth)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        depth = np.array(depth).reshape((depth.shape[0], depth.shape[1], depth.shape[2])).astype(np.float32) * 2.0
        augmented_image, augmented_depth, anomaly_mask, has_anomaly = self.augment_image(image, depth, anomaly_rgb=rgb_anomaly_source_list[anomaly_source_idx],
                                                                         anomaly_depth=depth_anomaly_source_list[anomaly_source_idx])
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        augmented_depth = np.transpose(augmented_depth, (2, 0, 1))
        # image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        # return image, augmented_image, anomaly_mask, has_anomaly
        # return torch.from_numpy(image), torch.from_numpy(augmented_image), torch.from_numpy(anomaly_mask), torch.from_numpy(has_anomaly)
        return torch.from_numpy(augmented_image), torch.from_numpy(augmented_depth[0,:,:]), torch.from_numpy(anomaly_mask), torch.from_numpy(has_anomaly)

    
    def __getitem__(self, idx):
        
        x, y, mask, xyz = self.x[idx], self.y[idx], self.mask[idx], self.xyz[idx]
        
        #TODO: add perlin noise, JIAQI! 
        if self.perlin:
            x, depth_map, mask, y = self.transform_image_perlin(x, xyz, resize_shape=self.resize_shape)
            x = x.unsqueeze()
            mask = mask.unsqueeze()
            depth_map = depth_map.unsqueeze()
        elif y == 0:
            x = Image.open(x).convert('RGB')
            x = self.imge_transform(x)
            mask = torch.zeros([1, x.shape[1], x.shape[2]])
            tiff_img = read_tiff(xyz)
            depth_map = tiff_to_depth(tiff=tiff_img, resized_img_size=self.data_transform['data_size'],
                                  duplicate=self.depth_duplicate)
        else:
            x = Image.open(x).convert('RGB')
            x = self.imge_transform(x)
            mask = Image.open(mask)
            mask = self.mask_transform(mask)
            tiff_img = read_tiff(xyz)
            depth_map = tiff_to_depth(tiff=tiff_img, resized_img_size=self.data_transform['data_size'],
                                  duplicate=self.depth_duplicate)
        
        y = y.unsqueeze()
        
         

        #return x, y, mask, depth_map, xyz  
        return {'rgb':x, 'label':y, 'gt_mask':mask, 'depth':depth_map, 'tiff': xyz} 

    def __len__(self):
        return len(self.x)


    def load_dataset(self):
        # input x, label y, [0, 1], good is 0 and bad is 1, mask is ground truth
        # train directory: only good cases
        # test directory: bad and good cases 
        # ground truth directory: only bad case

        for cls in self.class_names:
            img_dir = os.path.join(self.data_path, cls, self.phase)

            img_types = sorted(os.listdir(img_dir))
            for img_type in img_types:

                # load images
                img_type_dir = os.path.join(img_dir, img_type)
                if not os.path.isdir(img_type_dir):
                    continue
                img_path_list = sorted([os.path.join(img_type_dir, 'rgb', f)
                                        for f in os.listdir(img_type_dir + '/rgb')
                                        if f.endswith('.png')])
                self.x.extend(img_path_list)
                xyz_path_list = sorted([os.path.join(img_type_dir, 'xyz', f)
                                        for f in os.listdir(img_type_dir + '/xyz')
                                        if f.endswith('.tiff')])

                if img_type == 'good':
                    self.y.extend([0] * len(img_path_list))
                    self.mask.extend([None] * len(img_path_list))
                    # load xyz data
                    self.xyz.extend(xyz_path_list)
                else:
                    self.y.extend([1] * len(img_path_list))
                    gt_type_dir = os.path.join(img_dir, img_type, 'gt')
                    img_name_list = [os.path.splitext(os.path.basename(f))[0] for f in img_path_list]
                    gt_path_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                    for img_fname in img_name_list]
                    self.mask.extend(gt_path_list)
                    # load xyz data
                    xyz_type_dir = os.path.join(img_dir, img_type, 'xyz')
                    img_name_list = [os.path.splitext(os.path.basename(f))[0] for f in img_path_list]
                    xyz_path_list = [os.path.join(xyz_type_dir, img_fname + '.tiff')
                                    for img_fname in img_name_list]
                    self.xyz.extend(xyz_path_list)

        assert len(self.x) == len(self.y) == len(self.xyz), 'Number of Image Should Be The Same'
                        

class MVTecCL3D(Dataset):
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