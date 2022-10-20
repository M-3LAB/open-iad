#from torch.utils.data import  Dataset
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import numpy
import os
import os.path
import torch


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        class_label=list(class_to_idx.items())[class_index][0]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    if class_label=="good":
                        class_index=0
                    else:
                        class_index=1
                    item = path, class_index
                    instances.append(item)
    return instances


class mvtec_data(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/train/xxx.ext
        root/test/xxy.ext
        root/class_x/xxz.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions=None, classes=None,transform=None,
                 target_transform=None, is_valid_file=None,only_card=False,max_card=None,filter_pts=None,sample_max=None,sample_min=None):
        super(mvtec_data, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        self.root=os.path.join(root, classes[0])
        classes=[d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root,d))]
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.max_length=max_card
        self.loader = loader
        self.extensions = extensions
        self.only_card=only_card
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.filter_pts=filter_pts
        self.sample_max=sample_max
        self.sample_min= sample_min

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index,):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.only_card:
            path, target = self.samples[index]
            sample = self.loader(path)
            card = sample.shape[0]
            return card
        else:
            path, target = self.samples[index]
            if self.filter_pts is not None:
                sample = self.loader(path,pts_mean=self.filter_pts)
                if self.sample_min is not None:
                    sample=(sample-self.sample_min)/(self.sample_max-self.sample_min)
            else:
                sample = self.loader(path)
            if self.max_length is not None:
                desc_dim=sample.shape[-1]
                data=sample
                sample=torch.zeros(self.max_length,desc_dim)
                sample[0:data.shape[0]]=data
                card=data.shape[0]
            else:
                card=sample.shape[0]

            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            #return sample, target, card, path
            return sample, target, card

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)




