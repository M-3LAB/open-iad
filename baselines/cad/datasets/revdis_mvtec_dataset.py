from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from pathlib import Path
from torch.utils.data import Dataset

class RevDisTestMVTecDataset(Dataset):
    def __init__(self, root_dir, task_mvtec_classes, size):
        self.root_dir = root_dir
        self.task_mvtec_classes = task_mvtec_classes
        self.transform, self.gt_transform = self.get_data_transforms(size, size)
        # load dataset
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []

        for class_name in self.task_mvtec_classes:
            self.img_path = os.path.join(self.root_dir, class_name , "test")
            self.gt_path = os.path.join(self.root_dir, class_name, 'ground_truth')
            defect_types = os.listdir(self.img_path)

            for defect_type in defect_types:
                if defect_type == 'good':
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend([0] * len(img_paths))
                    tot_labels.extend([0] * len(img_paths))
                else:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                    gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                    img_paths.sort()
                    gt_paths.sort()
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend(gt_paths)
                    tot_labels.extend([1] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        return img_tot_paths, gt_tot_paths, tot_labels

    def get_data_transforms(self, size, isize):
        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]
        data_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.CenterCrop(isize),
            # transforms.CenterCrop(args.input_size),
            transforms.Normalize(mean=mean_train,
                                 std=std_train)])
        gt_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop(isize),
            transforms.ToTensor()])
        return data_transforms, gt_transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label