from torchvision import transforms
from .transforms.trans_cutpaste import CutPasteNormal, CutPasteScar, CutPaste3Way
from .transforms.maskimg import MaskImg
from .mvtec_dataset import MVTecAD
from .draem_mvtec_dataset import MVTecDRAEMTrainDataset, MVTecDRAEMTestDataset
from .revdis_mvtec_dataset import RevDisTestMVTecDataset
from torch.utils.data import DataLoader


def aug_transformation(args):
    if args.dataset.strong_augmentation:
        after_cutpaste_transform = transforms.Compose([
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        if args.dataset.random_aug:
            aug_transformation = transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.Resize((args.dataset.image_size, args.dataset.image_size)),
                transforms.RandomChoice([
                    CutPasteNormal(transform=after_cutpaste_transform),
                    CutPasteScar(transform=after_cutpaste_transform),
                    MaskImg(args.device, args.dataset.image_size, 0.25, [16, 2], colorJitter=0.1,
                            transform=after_cutpaste_transform)])
            ])
        else:
            aug_transformation = transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.Resize((args.dataset.image_size, args.dataset.image_size)),
                # MaskImg(args.device, args.dataset.image_size, 0.25, [16, 2], colorJitter=0.1, transform=after_cutpaste_transform)
                CutPasteNormal(transform=after_cutpaste_transform)
            ])
    else:
        aug_transformation = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.Resize(args.dataset.image_size),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return aug_transformation

def no_aug_transformation(args):
    no_aug_transformation = transforms.Compose([
        transforms.Resize((args.dataset.image_size, args.dataset.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return no_aug_transformation

def get_mvtec_dataloaders(args, t, dataloaders_train, dataloaders_test, learned_tasks):
    if args.dataset.dataset_order == 1:
        mvtec_classes = ['leather', 'bottle', 'metal_nut',
                         'grid', 'screw', 'zipper',
                         'tile', 'hazelnut', 'toothbrush',
                         'wood', 'transistor', 'pill',
                         'carpet', 'capsule', 'cable']
    elif args.dataset.dataset_order == 2:
        mvtec_classes = ['wood', 'transistor', 'pill',
                         'tile', 'hazelnut', 'toothbrush',
                         'leather', 'bottle', 'metal_nut',
                         'carpet', 'capsule', 'cable',
                         'grid', 'screw', 'zipper']
    elif args.dataset.dataset_order == 3:
        mvtec_classes = ['leather', 'grid', 'tile',
                         'bottle', 'toothbrush', 'capsule',
                         'screw', 'pill', 'zipper',
                         'cable', 'metal_nut', 'hazelnut',
                         'wood', 'carpet', 'transistor']

    N_CLASSES_PER_TASK = args.dataset.n_classes_per_task

    if args.dataset.data_incre_setting == 'one':
        # N_CLASSES_PER_TASK = 1
        if t == 0:
            task_mvtec_classes = mvtec_classes[: 10]
        else:
            i = 10 + (t - 1) * N_CLASSES_PER_TASK
            task_mvtec_classes = mvtec_classes[i: i + N_CLASSES_PER_TASK]
    else:
        # N_CLASSES_PER_TASK = 3
        i = t * N_CLASSES_PER_TASK
        task_mvtec_classes = mvtec_classes[i: i + N_CLASSES_PER_TASK]
    learned_tasks.append(task_mvtec_classes)

    train_transform = aug_transformation(args)
    test_transform = no_aug_transformation(args)

    if args.model.name == 'draem':
        train_data = MVTecDRAEMTrainDataset(args.data_dir, args.anomaly_source_path,
                                            task_mvtec_classes, resize_shape=[args.dataset.image_size, args.dataset.image_size])
        test_data = MVTecDRAEMTestDataset(args.data_dir, task_mvtec_classes, resize_shape=[args.dataset.image_size, args.dataset.image_size])
    elif args.model.name == 'revdis':
        train_data = MVTecAD(args.data_dir, task_mvtec_classes, transform=test_transform, size=args.dataset.image_size)
        test_data = RevDisTestMVTecDataset(args.data_dir, task_mvtec_classes, size=args.dataset.image_size)
    else:
        train_data = MVTecAD(args.data_dir, task_mvtec_classes,
                             transform=train_transform, size=args.dataset.image_size)
        test_data = MVTecAD(args.data_dir, task_mvtec_classes, args.dataset.image_size,
                            transform=test_transform, mode="test")

    train_dataloader = DataLoader(train_data, batch_size=args.train.batch_size, shuffle=True, num_workers=args.dataset.num_workers)
    dataloaders_train.append(train_dataloader)
    dataloader_test = DataLoader(test_data, batch_size=args.eval.batch_size, shuffle=False, num_workers=args.dataset.num_workers)
    dataloaders_test.append(dataloader_test)
    print('class name:', task_mvtec_classes, 'number of training sets:', len(train_data),
          'number of testing sets:', len(test_data))

    return train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, len(train_data)