import numpy as np
import os, sys
from datasets.base_dataset import BaseADDataset
from PIL import Image
from torchvision import transforms

class MVTecAD(BaseADDataset):

    def __init__(self, args, train = True):
        super(MVTecAD).__init__()
        self.args = args
        self.train = train
        self.classname = self.args.classname

        self.root = os.path.join(self.args.dataset_root, self.classname)
        self.transform = self.transform_train() if self.train else self.transform_test()

        normal_data = list()
        if self.train:
            split = 'train'
        else:
            split = 'test'
        normal_files = os.listdir(os.path.join(self.root, split, 'good'))
        for file in normal_files:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                normal_data.append(split + '/good/' + file)

        outlier_data = self.split_outlier()
        outlier_data.sort()

        normal_label = np.zeros(len(normal_data)).tolist()
        outlier_label = np.ones(len(outlier_data)).tolist()

        self.images = normal_data + outlier_data
        self.labels = np.array(normal_label + outlier_label)
        self.normal_idx = np.argwhere(self.labels == 0).flatten()
        self.outlier_idx = np.argwhere(self.labels == 1).flatten()


    def split_outlier(self):
        outlier_data_dir = os.path.join(self.root, 'test')
        outlier_classes = os.listdir(outlier_data_dir)
        outlier_data = list()
        for cl in outlier_classes:
            if cl == 'good':
                continue
            outlier_file = os.listdir(os.path.join(outlier_data_dir, cl))
            for file in outlier_file:
                if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                    outlier_data.append('test/' + cl + '/' + file)
        if self.args.n_anomaly > len(outlier_data)/2:
            print(len(outlier_data))
            print("Number of outlier data in training set should less than half of outlier datas!")
            sys.exit()
        np.random.RandomState(self.args.ramdn_seed).shuffle(outlier_data)
        if self.train:
            return outlier_data[0:self.args.n_anomaly]
        else:
            return outlier_data[self.args.n_anomaly:]

    def load_image(self, path):
        if 'npy' in path[-3:]:
            img = np.load(path).astype(np.uint8)
            img = img[:, :, :3]
            return Image.fromarray(img)
        return Image.open(path).convert('RGB')

    def transform_train(self):
        composed_transforms = transforms.Compose([
            transforms.Resize((self.args.img_size,self.args.img_size)),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return composed_transforms

    def transform_test(self):
        composed_transforms = transforms.Compose([
            transforms.Resize((self.args.img_size, self.args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return composed_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        transform = self.transform
        image = self.load_image(os.path.join(self.root, self.images[index]))
        sample = {'image': transform(image), 'label': self.labels[index]}
        return sample

    def getitem(self, index):
        if index in self.outlier_idx and self.train:
            transform = self.transform_anomaly
        else:
            transform = self.transform
        image = self.load_image(os.path.join(self.root, self.images[index]))
        if index in self.outlier_idx:
            image_label = self.load_image(os.path.join(self.root,self.images[index]).replace('test','ground_truth').replace('.png','_mask.png'))
        else:
            image_label = None
        sample = {'image': transform(image), 'label': self.labels[index], 'seg_label': image_label, 'raw_image':image}
        return sample
