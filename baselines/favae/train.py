import os
import random
import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models
from torchvision.utils import save_image
from tqdm import tqdm
from datasets.mvtec import MVTecDataset
from datasets.preprocessing import generate_image_list, augment_images
from func import feature_extractor, EarlyStop
from utils import time_file_str, time_string, convert_secs2time, print_log, AverageMeter
from models.VAE import VAE

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def main():
    parser = argparse.ArgumentParser(description='FAVAE anomaly detection')
    parser.add_argument('--obj', type=str, default='.')
    parser.add_argument('--data_type', type=str, default='mvtec')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--epochs', type=int, default=100, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--validation_ratio', type=float, default=0.2)
    parser.add_argument('--grayscale', action='store_true', help='color or grayscale input image')
    parser.add_argument('--img_resize', type=int, default=128)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--do_aug', action='store_true', help='whether to do data augmentation before training')
    parser.add_argument('--augment_num', type=int, default=10000)
    parser.add_argument('--p_rotate', type=float, default=0.3, help='probability to do image rotation')
    parser.add_argument('--rotate_angle_vari', type=float, default=15.0, help='rotate image between [-angle, +angle]')
    parser.add_argument('--p_rotate_crop', type=float, default=1.0, help='probability to crop inner rotated image')
    parser.add_argument('--p_horizonal_flip', type=float, default=0.3, help='probability to do horizonal flip')
    parser.add_argument('--p_vertical_flip', type=float, default=0.3, help='probability to do vertical flip')
    parser.add_argument('--kld_weight', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate of Adam')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='decay of Adam')
    parser.add_argument('--seed', type=int, default=None, help='manual seed')
    args = parser.parse_args()

    args.p_crop = 1 if args.crop_size != args.img_resize else 0
    args.train_data_dir = args.data_path + '/' + args.obj + '/train/good'
    args.aug_dir = './train_patches/' + args.obj + '/train/good'

    args.input_channel = 1 if args.grayscale else 3

    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    args.prefix = time_file_str()
    args.save_dir = './' + args.data_type + '/' + args.obj + '/vgg_feature' + '/seed_{}/'.format(args.seed)
    
    # data augmentation
    if not os.path.exists(args.aug_dir) and args.do_aug:
        os.makedirs(args.aug_dir)
        img_list = generate_image_list(args)
        augment_images(img_list, args)
    
    args.train_data_path = './train_patches'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, 'model_training_log_{}.txt'.format(args.prefix)), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    # load model and dataset
    model = VAE(input_channel=args.input_channel, z_dim=100).to(device)
    teacher = models.vgg16(pretrained=True).to(device)
    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    img_size = args.crop_size if args.img_resize != args.crop_size else args.img_resize
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = MVTecDataset(args.train_data_path, class_name=args.obj, is_train=True, resize=img_size)
    img_nums = len(train_dataset)
    valid_num = int(img_nums * args.validation_ratio)
    train_num = img_nums - valid_num
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_num, valid_num])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False, **kwargs)

    test_dataset = MVTecDataset(args.data_path, class_name=args.obj, is_train=False, resize=img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, **kwargs)

    # fetch fixed data for debugging
    x_normal_fixed, _, _ = iter(val_loader).next()
    x_normal_fixed = x_normal_fixed.to(device)

    x_test_fixed, _, _ = iter(test_loader).next()
    x_test_fixed = x_test_fixed.to(device)

    # start training
    save_name = os.path.join(args.save_dir, '{}_{}_model.pt'.format(args.obj, args.prefix))
    early_stop = EarlyStop(patience=20, save_name=save_name)
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(1, args.epochs + 1):
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time), log)
        train(args, model, teacher, epoch, train_loader, optimizer, log)
        val_loss = val(args, model, teacher, epoch, val_loader, log)

        if (early_stop(val_loss, model, optimizer, log)):
            break

        if epoch % 10 == 0:
            save_sample = os.path.join(args.save_dir, '{}val-images.jpg'.format(epoch))
            save_sample2 = os.path.join(args.save_dir, '{}test-images.jpg'.format(epoch))
            save_snapshot(x_normal_fixed, x_test_fixed, model, save_sample, save_sample2, log)

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
    log.close()


def train(args, model, teacher, epoch, train_loader, optimizer, log):
    model.train()
    teacher.eval()
    losses = AverageMeter()
    MSE_loss = nn.MSELoss(reduction='sum')

    for (data, _, _) in tqdm(train_loader):
        data = data.to(device)
        z, output, mu, log_var = model(data)
        # get model's intermediate outputs
        s_activations, _ = feature_extractor(z, model.decode, target_layers=['10', '16', '22'])
        t_activations, _ = feature_extractor(data, teacher.features, target_layers=['7', '14', '21'])

        optimizer.zero_grad()
        mse_loss = MSE_loss(output, data)
        kld_loss = 0.5 * torch.sum(-1 - log_var + torch.exp(log_var) + mu**2)
        for i in range(len(s_activations)):
            s_act = model.adapter[i](s_activations[-(i + 1)])
            mse_loss += MSE_loss(s_act, t_activations[i])
        loss = mse_loss + args.kld_weight * kld_loss
        losses.update(loss.sum().item(), data.size(0))
 
        loss.backward()
        optimizer.step()

    print_log(('Train Epoch: {} Loss: {:.6f}'.format(epoch, losses.avg)), log)


def val(args, model, teacher, epoch, val_loader, log):
    model.eval()
    teacher.eval()
    losses = AverageMeter()
    MSE_loss = nn.MSELoss(reduction='sum')
    for (data, _, _) in tqdm(val_loader):
        data = data.to(device)
        with torch.no_grad():
            z, output, mu, log_var = model(data)
            # get model's intermediate outputs
            s_activations, _ = feature_extractor(z, model.decode, target_layers=['12', '18', '24'])
            t_activations, _ = feature_extractor(data, teacher.features, target_layers=['8', '15', '22'])

            mse_loss = MSE_loss(output, data)
            kld_loss = 0.5 * torch.sum(-1 - log_var + torch.exp(log_var) + mu**2)
            for i in range(len(s_activations)):
                s_act = model.adapter[i](s_activations[-(i + 1)])
                mse_loss += MSE_loss(s_act, t_activations[i])
            loss = mse_loss + args.kld_weight * kld_loss
            losses.update(loss.item(), data.size(0))

    print_log(('Val Epoch: {} loss: {:.6f}'.format(epoch, losses.avg)), log)

    return losses.avg


def save_snapshot(x, x2, model, save_dir, save_dir2, log):
    model.eval()
    with torch.no_grad():
        x_fake_list = x
        recon = model(x)[1]
        x_concat = torch.cat((x_fake_list, recon), dim=3)
        save_image(x_concat.data.cpu(), save_dir, nrow=1, padding=0)
        print_log(('Saved real and fake images into {}...'.format(save_dir)), log)

        x_fake_list = x2
        recon = model(x2)[1]
        x_concat = torch.cat((x_fake_list, recon), dim=3)
        save_image(x_concat.data.cpu(), save_dir2, nrow=1, padding=0)
        print_log(('Saved real and fake images into {}...'.format(save_dir2)), log)


if __name__ == '__main__':
    main()
