from torchvision import transforms as T
from cutpaste_augmentation import *

__all__ = ['total_aug']

def total_aug(args):
    if args['augment_type'] == 'normal':
        imge_transform = T.Compose([T.Resize((self.data_transform['data_size'], self.data_transform['data_size'])),
                                    T.CenterCrop(self.data_transform['data_crop_size']),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

        mask_transform = T.Compose([T.Resize(self.data_transform['mask_size']),
                                        T.CenterCrop(self.data_transform['mask_crop_size']),
                                        T.ToTensor(),
                                        ])
