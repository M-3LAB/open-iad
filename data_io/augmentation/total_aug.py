from torchvision import transforms
from cutpaste_augmentation import *

__all__ = ['total_aug']

def total_aug(args):
    if args['augment_type'] == 'normal':
        pass
