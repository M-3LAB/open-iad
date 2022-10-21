from torchvision import transforms as T
from cutpaste_augmentation import *

__all__ = ['total_aug']

def total_aug(args):
    if args['augment_type'] == 'normal':
        img_transform = T.Compose([T.Resize((args['data_size'], args['data_size'])),
                                    T.CenterCrop(args['data_crop_size']),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

        mask_transform = T.Compose([T.Resize(args['mask_size']),
                                    T.CenterCrop(args['mask_crop_size']),
                                    T.ToTensor(),
                                    ])
    
    elif args['augment_type'] == 'cutpaste':
        #after_cutpaste_transform = 
        pass
    
    
    return img_transform, mask_transform
