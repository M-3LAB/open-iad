import numpy as np
import os, sys
from datasets.base_dataset import BaseADDataset
from PIL import Image
from torchvision import transforms
from datasets.cutmix import CutMix
import random

class MVTecAD(BaseADDataset):

    def __init__(self, args, train = True):
        super(MVTecAD).__init__()
        self.args = args
        self.train = train
        self.classname = self.args.classname
        self.know_class = self.args.know_class
        self.pollution_rate = self.args.cont_rate
        if self.args.test_threshold == 0 and self.args.test_rate == 0:
            self.test_threshold = self.args.nAnomaly
        else:
            self.test_threshold = self.args.test_threshold

        self.root = os.path.join(self.args.dataset_root, self.classname)
        self.transform = self.transform_train() if self.train else self.transform_test()
        self.transform_pseudo = self.transform_pseudo()

        normal_data = list()
        split = 'train'
        normal_files = os.listdir(os.path.join(self.root, split, 'good'))
        for file in normal_files:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                normal_data.append(split + '/good/' + file)

        self.nPollution = int((len(normal_data)/(1-self.pollution_rate)) * self.pollution_rate)
        if self.test_threshold==0 and self.args.test_rate>0:
            self.test_threshold = int((len(normal_data)/(1-self.args.test_rate)) * self.args.test_rate) + self.args.nAnomaly

        self.ood_data = self.get_ood_data()

        if self.train is False:
            normal_data = list()
            split = 'test'
            normal_files = os.listdir(os.path.join(self.root, split, 'good'))
            for file in normal_files:
                if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                    normal_data.append(split + '/good/' + file)

        outlier_data, pollution_data = self.split_outlier()
        outlier_data.sort()

        normal_data = normal_data + pollution_data

        normal_label = np.zeros(len(normal_data)).tolist()
        outlier_label = np.ones(len(outlier_data)).tolist()

        self.images = normal_data + outlier_data
        self.labels = np.array(normal_label + outlier_label)
        self.normal_idx = np.argwhere(self.labels == 0).flatten()
        self.outlier_idx = np.argwhere(self.labels == 1).flatten()

    def get_ood_data(self):
        ood_data = list()
        if self.args.outlier_root is None:
            return None
        dataset_classes = os.listdir(self.args.outlier_root)
        for cl in dataset_classes:
            if cl == self.args.classname:
                continue
            cl_root = os.path.join(self.args.outlier_root, cl, 'train', 'good')
            ood_file = os.listdir(cl_root)
            for file in ood_file:
                if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                    ood_data.append(os.path.join(cl_root, file))
        return ood_data

    def split_outlier(self):
        outlier_data_dir = os.path.join(self.root, 'test')
        outlier_classes = os.listdir(outlier_data_dir)
        if self.know_class in outlier_classes:
            print("Know outlier class: " + self.know_class)
            outlier_data = list()
            know_class_data = list()
            for cl in outlier_classes:
                if cl == 'good':
                    continue
                outlier_file = os.listdir(os.path.join(outlier_data_dir, cl))
                for file in outlier_file:
                    if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                        if cl == self.know_class:
                            know_class_data.append('test/' + cl + '/' + file)
                        else:
                            outlier_data.append('test/' + cl + '/' + file)
            np.random.RandomState(self.args.ramdn_seed).shuffle(know_class_data)
            know_outlier = know_class_data[0:self.args.nAnomaly]
            unknow_outlier = outlier_data
            if self.train:
                return know_outlier, list()
            else:
                return unknow_outlier, list()


        outlier_data = list()
        for cl in outlier_classes:
            if cl == 'good':
                continue
            outlier_file = os.listdir(os.path.join(outlier_data_dir, cl))
            for file in outlier_file:
                if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                    outlier_data.append('test/' + cl + '/' + file)
        np.random.RandomState(self.args.ramdn_seed).shuffle(outlier_data)
        if self.train:
            return outlier_data[0:self.args.nAnomaly], outlier_data[self.args.nAnomaly:self.args.nAnomaly + self.nPollution]
        else:
            return outlier_data[self.test_threshold:], list()

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

    def transform_pseudo(self):
        composed_transforms = transforms.Compose([
            transforms.Resize((self.args.img_size,self.args.img_size)),
            CutMix(),
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
        rnd = random.randint(0, 1)
        if index in self.normal_idx and rnd == 0 and self.train:
            if self.ood_data is None:
                index = random.choice(self.normal_idx)
                image = self.load_image(os.path.join(self.root, self.images[index]))
                transform = self.transform_pseudo
            else:
                image = self.load_image(random.choice(self.ood_data))
                transform = self.transform
            label = 2
        else:
            image = self.load_image(os.path.join(self.root, self.images[index]))
            transform = self.transform
            label = self.labels[index]
        sample = {'image': transform(image), 'label': label}
        return sample
