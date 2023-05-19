from torchvision import transforms as T
from augmentation.cutpaste_aug import *

__all__ = ['aug_type']

def aug_type(augment_type, args):
    if augment_type == 'normal':
        img_transform = T.Compose([T.Resize((args['data_size'], args['data_size'])),
                                    T.CenterCrop(args['data_crop_size']),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

        mask_transform = T.Compose([T.Resize(args['mask_size']),
                                    T.CenterCrop(args['mask_crop_size']),
                                    T.ToTensor(),
                                    ])
    
    elif augment_type == 'cutpaste':
        after_cutpaste_transform = T.Compose([T.RandomRotation(90),
                                              T.ToTensor(),
                                              T.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                            ])
        
        img_transform = T.Compose([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                   T.Resize((args['data_crop_size'], args['data_crop_size'])),
                                   CutPasteNormal(transform=after_cutpaste_transform)
                                   #T.RandomChoice([CutPasteNormal(transform=after_cutpaste_transform),
                                   #                CutPasteScar(transform=after_cutpaste_transform)])
                                   ])

        mask_transform = T.Compose([T.Resize(args['mask_size']),
                                    T.CenterCrop(args['mask_crop_size']),
                                    T.ToTensor(),
                                    ])
    else:
        raise NotImplementedError('The Augmentation Type Has Not Been Implemented Yet')

    
    
    return img_transform, mask_transform
