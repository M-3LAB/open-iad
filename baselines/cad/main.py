from tqdm import tqdm
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import numpy as np
from eval import eval_model
import timm
from arguments import get_args
from models import ResNetModel, ViT, Dis, Discat, PANDA, CutPaste, CSFlow, NetCSFlow, DRAEM, NetDRAEM, RevDis, NetRevDis, CLFlow, NetCLFlow
from datasets import get_mvtec_dataloaders, get_mtd_mvtec_dataloaders, get_joint_mtd_mvtec_dataloaders
from utils.optimizer import get_optimizer
from utils.density import GaussianDensityTorch


def compactness_loss(device, net, train_loader):
    net.eval()
    train_feature_space = []
    with torch.no_grad():
        for imgs in train_loader:
            imgs = imgs.to(device)
            features = net.forward_features(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    center = torch.FloatTensor(train_feature_space).mean(dim=0)
    return center

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def main(args):
    # create Model:
    if args.model.net == 'resnet':
        head_layers = [512] * 2 + [128]
        net = ResNetModel(pretrained=args.model.pretrained, head_layers=head_layers, num_classes=args.train.num_classes)
        optimizer = get_optimizer(args, net)
        scheduler = CosineAnnealingWarmRestarts(optimizer, args.train.num_epochs)
    elif args.model.net == 'net_clflow':
        head_layers = [512] * 2 + [128]
        net = NetCLFlow(args, head_layers)
        optimizer = get_optimizer(args, net)
        scheduler = CosineAnnealingWarmRestarts(optimizer, args.train.num_epochs)
    elif args.model.net == 'vit':
        net = ViT(num_classes=args.train.num_classes)
        if args.model.pretrained:
            checkpoint_path = './checkpoints/sam-ViT-B_16.npz'
            net.load_pretrained(checkpoint_path)
        optimizer = get_optimizer(args, net)
        scheduler = CosineAnnealingWarmRestarts(optimizer, args.train.num_epochs)
    elif args.model.net == 'net_csflow':
        net = NetCSFlow(args)
        optimizer = get_optimizer(args, net)
        scheduler = None
    elif args.model.net == 'net_draem':
        net = NetDRAEM(args)
        net.apply(weights_init)
        optimizer = get_optimizer(args, net)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.train.num_epochs * 0.8, args.train.num_epochs * 0.9], gamma=0.2, last_epoch=-1)
    elif args.model.net == 'net_revdis':
        net = NetRevDis(args)
        optimizer = get_optimizer(args, net)
        scheduler = None

    net.to(args.device)

    if args.model.name == 'dis':
        model = Dis(args, net, optimizer, scheduler)
    elif args.model.name == 'discat':
        model = Discat(args, net, optimizer, scheduler)
    elif args.model.name == 'panda':
        model = PANDA(args, net, optimizer, scheduler)
        args.dataset.strong_augmentation = False
    elif args.model.name == 'cutpaste':
        model = CutPaste(args, net, optimizer, scheduler)
        args.dataset.strong_augmentation = True
    elif args.model.name == 'upper':
        model = CutPaste(args, net, optimizer, scheduler)
        args.dataset.strong_augmentation = True
    elif args.model.name == 'csflow':
        model = CSFlow(args, net, optimizer, scheduler)
        args.dataset.strong_augmentation = False
    elif args.model.name == 'clflow':
        model = CLFlow(args, net, optimizer, scheduler)
        args.dataset.strong_augmentation = False
    elif args.model.name == 'draem':
        model = DRAEM(args, net, optimizer, scheduler)
        args.dataset.strong_augmentation = False
    elif args.model.name == 'revdis':
        model = RevDis(args, net, optimizer, scheduler)
        args.dataset.strong_augmentation = False

    density = GaussianDensityTorch()

    dataloaders_train, dataloaders_test, learned_tasks = [], [], []
    task_wise_mean, task_wise_cov, task_wise_train_data_nums = [], [], []

    for t in range(args.dataset.n_tasks):
        print('---' * 10, f'Task:{t}', '---' * 10)
        if args.dataset.name == 'seq-mvtec':
            train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, data_train_nums = get_mvtec_dataloaders(args, t, dataloaders_train, dataloaders_test, learned_tasks)
        elif args.dataset.name == 'seq-mtd-mvtec':
            train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, data_train_nums = get_mtd_mvtec_dataloaders(args, t, dataloaders_train, dataloaders_test, learned_tasks)
        elif args.dataset.name == 'joint-mtd-mvtec':
            train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, data_train_nums = get_joint_mtd_mvtec_dataloaders(args, dataloaders_train, dataloaders_test, learned_tasks)

        task_wise_train_data_nums.append(data_train_nums)

        if args.model.name == 'panda':
            center = compactness_loss(args.device, net, train_dataloader)

        if args.model.name == 'csflow':
            net.feature_extractor.eval()
            net.density_estimator.train()
        elif args.model.name == 'revdis':
            net.encoder.eval()
            net.decoder.train()
            net.bn.train()
        else:
            net.train()

        for epoch in tqdm(range(args.train.num_epochs)):
            one_epoch_embeds = []

            if args.model.name == 'upper':
                for dataloader_train in dataloaders_train:
                    for batch_idx, (data) in enumerate(dataloader_train):
                        if isinstance(data, list):
                            inputs = [x.to(args.device) for x in data]
                            labels = torch.arange(len(inputs), device=args.device)
                            labels = labels.repeat_interleave(inputs[0].size(0))
                            inputs = torch.cat(inputs, dim=0)
                        else:
                            inputs = data.to(args.device)
                            labels = torch.zeros(inputs.size(0), device=args.device)

                        model(epoch, inputs, labels, one_epoch_embeds, task_wise_mean, task_wise_cov, t)
            else:
                for batch_idx, (data) in enumerate(train_dataloader):
                    if args.model.name == 'draem':
                        inputs = [x.to(args.device) for x in data[:-2]]
                        masks = data[-2].to(args.device)
                        labels = data[-1].to(args.device)
                        inputs, labels = torch.cat(inputs, dim=0), torch.squeeze(labels, 1)
                    else:
                        if isinstance(data, list):
                            inputs = [x.to(args.device) for x in data]
                            labels = torch.arange(len(inputs), device=args.device)
                            labels = labels.repeat_interleave(inputs[0].size(0))
                            inputs = torch.cat(inputs, dim=0)
                        else:
                            inputs = data.to(args.device)
                            labels = torch.zeros(inputs.size(0), device=args.device)

                    if args.model.name == 'panda':
                        model(epoch, inputs, one_epoch_embeds, center, t)
                    elif args.model.name == 'draem':
                        model(epoch, inputs, labels, masks)
                    else:
                        model(epoch, inputs, labels, one_epoch_embeds, task_wise_mean, task_wise_cov, t)

            if args.model.name == 'draem':
                scheduler.step()

            if args.train.test_epochs > 0 and epoch % args.train.test_epochs == 0:
                net.eval()
                if args.model.name == 'discat':
                    density, task_wise_mean, task_wise_cov = model.training_epoch(density, one_epoch_embeds, task_wise_mean, task_wise_cov, task_wise_train_data_nums, t)
                else:
                    density = model.training_epoch(density, one_epoch_embeds, task_wise_mean, task_wise_cov, task_wise_train_data_nums, t)

                eval_model(args, epoch, dataloaders_test, learned_tasks, net, density)
    torch.save(net, f'checkpoints/{args.model.name}{args.seed}_{args.dataset.name}{args.dataset.data_incre_setting}_epochs{args.train.num_epochs}.pth')
    torch.save(density, f'checkpoints/density_{args.model.name}{args.seed}_{args.dataset.name}{args.dataset.data_incre_setting}_epochs{args.train.num_epochs}.pth')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    args = get_args()
    main(args)