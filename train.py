import torch
import torch.nn as nn
from configuration.config import parse_argument 
from data_io.augmentation import *
from data_io.mvtec_ad import *
from data_io.mtd import *
from torch.utils.data import DataLoader


if __name__ == "__main__":
    args = parse_argument()

    if args.all_classes:
        if args.dataset == 'mvtec2d':
            class_name = mvtec_2d_classes()
        elif args.dataset == 'mvtec3d':
            class_name = mvtec_3d_classes()
        elif args.dataset == 'mtd':
            class_name = mtd_classes() 
    else:
        class_name = args.class_name

    if args.dataset == 'mvtec2d':
        if args.mode == 'centralized':
            mvtec_2d_trainset = MVTec2D(data_path=args.data_path, class_name=class_name,
                                        phase='train', mode=args.mode, 
                                        data_transform=mvtec_2d_image_transform,
                                        mask_transform=mvtec_2d_mask_transform)

            mvtec_2d_testset = MVTec2D(data_path=args.data_path, class_name=class_name,
                                    phase='test', mode=args.mode,
                                    data_transform=mvtec_2d_image_transform,
                                    mask_transform=mvtec_2d_mask_transform)
            
            train_loader = DataLoader(mvtec_2d_trainset, batch_size=args.batchsize,
                                    shuffle=True, num_workers=args.num_workers) 

            test_loader = DataLoader(mvtec_2d_testset, batch_size=args.batchsize,
                                    shuffle=False, num_workers=0) 

        elif args.mode == 'continual':
            mvtec_2d_trainsetlist = MVTec2DContinualList(args.data_path, class_name, args.num_tasks_continual, 'train', args.mode, mvtec_2d_image_transform, mvtec_2d_mask_transform)
            mvtec_2d_testsetlist = MVTec2DContinualList(args.data_path, class_name, args.num_tasks_continual, 'test', args.mode, mvtec_2d_image_transform, mvtec_2d_mask_transform)
            train_loaderList = MVTec2DContinualDataloaderList(mvtec_2d_trainsetlist, args.batchsize, False, args.num_workers)
            test_loaderList = MVTec2DContinualDataloaderList(mvtec_2d_testsetlist, args.batchsize, False, args.num_workers)
            #error appears when shuffle is True, ValueError: sampler option is mutually exclusive with shuffle
            # for i in train_loaderList:
            #     print(i.__len__)


    elif args.dataset == 'mvtec3d':
        mvtec_3d_trainset = MVTec3D(phase='train')
        mvtec_3d_testset = MVTec3D(phase='test')

    elif args.dataset == 'mtd':
        mtd_trainset = MTD(phase='train') 
        mtd_testset = MTD(phase='test')

    else:
        raise NotImplementedError('This Dataset Have Not Been Implemented Yet')