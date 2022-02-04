import torch
import torch.nn as nn
from configuration.config import parse_argument 
from data_io.augmentation import *
from data_io.mvtec_ad import *

if __name__ == "__main__":
    args = parse_argument()
    if args.all_classes:
        class_name = mvtec_2d_classes()
    else:
        class_name = args.class_name
    mvtec_2d_trainset = MVTec2D(data_path=args.data_path)