import os
import numpy
from Recorder import Recorder
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import autograd
from torch import optim
import Helper
import torch.nn.init as init
from timeit import default_timer as timer
import cv2
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from pytorch_msssim import ms_ssim, ssim
from tqdm import tqdm
import numpy as np
from p32.ssim_module import *
from torch.autograd import Variable
from p32.mvtex_data_loader import *
import torchvision.transforms.functional as TF

import argparse
device = torch.device("cuda:0")
print(">> Device Info: {} is in use".format(device))
parser = argparse.ArgumentParser(description='CIFAR10 Training')
parser.add_argument('-n', '--num', nargs='+', type=int, help='<Required> Set flag', required=True)
parser.add_argument('-sr', '--sample_rate', default=1, type=float)


device = torch.device("cuda:0")
print(">> Device Info: {} is in use".format(device))

DIM = 32                    # Model dimensionality
CRITIC_ITERS = 5            # How many iterations to train the critic for
GENER_ITERS = 1
N_GPUS = 1                  # Number of GPUs
BATCH_SIZE = 2            # Batch size. Must be a multiple of N_GPUS
global END_ITER
MAX_EPOCH = 256
LAMBDA = 10                 # Gradient pena1lty lambda hyperparameter
OUTPUT_DIM = 32 * 32 * 3    # Number of pixels in each image
# Stride to extract patches
p_stride = 16
# global BEST_AUC
BEST_AUC = 0
############################ Parameters ############################
latent_dimension = 128

category = {
            1: "bottle",
            2: "hazelnut",
            3: "capsule",
            4: "metal_nut",
            5: "leather",
            6: "pill",
            7: "wood",
            8: "carpet",
            9: "tile",
            10: "grid",
            11: "cable",
            12: "transistor",
            13: "toothbrush",
            14: "screw",
            15: "zipper"
            }

data_range = 2.1179 + 2.6400
ssim_weights = [0.0516, 0.3295, 0.3463, 0.2726]

sig_f = 1
rec_f = 10
svd_f = 0.1

####################################################################
num_worker = 12
USE_SSIM = True

LR = 1e-4       # 0.0001
mean_dist = None

mse_criterion = torch.nn.MSELoss()
l1_criterion = torch.nn.L1Loss()
bce_criterion = torch.nn.BCELoss()

sigbce_criterion = torch.nn.BCEWithLogitsLoss()

def create_dir(dir):
    if not os.path.exists(dir):
            os.mkdir(path=dir)


def weights_init(m):
    if isinstance(m, MyConvo2d):
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, torch.nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)


mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]


def load_train(train_path, sample_rate):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    imagenet_data = MvtecDataLoader(train_path, transform=transform, mode="train", sample_rate=sample_rate)
    # imagenet_data = torchvision.datasets.ImageFolder(train_path, transform=transform)

    train_data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=num_worker,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    return train_data_loader, imagenet_data.__len__()


def load_test(test_path, sample_rate):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    imagenet_data = MvtecDataLoader(test_path, transform=transform, mode="test", sample_rate=sample_rate)

    valid_data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=num_worker,
                                                    pin_memory=True,
                                                    shuffle=False)
    return valid_data_loader


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=0, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
    if max_iter == 0:
        raise Exception("MAX ITERATION CANNOT BE ZERO!")
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer
    lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def extract_patch(data_tmp):
    tmp = None
    _, _, a, b, _, _ = data_tmp.shape
    for i in range(a):
        for j in range(b):
            tmp = data_tmp[:, :, i, j, :, :] if i == 0 and j == 0 \
                else torch.cat((tmp, data_tmp[:, :, i, j, :, :]), dim=0)
    return tmp


def init_c(DataLoader, net, eps=0.1):
    c = torch.zeros((1, latent_dimension)).to(device)
    net.eval()
    n_samples = 0
    print("Estimating Center ...")
    with torch.no_grad():
        for index, (images, label) in enumerate(tqdm(DataLoader, position=0)):
            img_org = images.to(device)
            img_tmp = img_org.unfold(2, 32, p_stride).unfold(3, 32, p_stride)
            img_tmp = img_tmp.permute(0, 2, 3, 1, 4, 5)
            img = img_tmp.reshape(-1, 3, 32, 32)
            outputs = net.encoder(img)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)
    c /= n_samples
    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c


def init_sigma(DataLoader, net):
    net.eval()
    tmp_sigma = torch.tensor(0.0, dtype=torch.float).to(device)
    n_samples = 0
    print("Estimating Standard Deviation ...")
    with torch.no_grad():
        for index, (images, label) in enumerate(tqdm(DataLoader, position=0)):
            img_org = images.to(device)
            img_tmp = img_org.unfold(2, 32, p_stride).unfold(3, 32, p_stride)
            img_tmp = img_tmp.permute(0, 2, 3, 1, 4, 5)
            img = img_tmp.reshape(-1, 3, 32, 32)
            latent_z = net.encoder(img)
            diff = (latent_z - generator.c) ** 2
            tmp = torch.sum(diff.detach(), dim=1)
            if (tmp.mean().detach() / sig_f) < 1:
                tmp_sigma += 1
            else:
                tmp_sigma += tmp.mean().detach() / sig_f
            n_samples += 1
    tmp_sigma /= n_samples
    return tmp_sigma


def train(args, NORMAL_NUM, generator, discriminator, optimizer_g, optimizer_d):
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    generator.train()
    discriminator.train()

    AUC_LIST = []
    global test_auc
    test_auc = 0
    BEST_AUC = 0
    train_path = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/train/good'.format(NORMAL_NUM)
    START_ITER = 0
    train_size = len(os.listdir(train_path))
    END_ITER = int((train_size / BATCH_SIZE) * MAX_EPOCH)
    # END_ITER = 40500

    train_dataset_loader, train_size = load_train(train_path, args.sample_rate)

    generator.c = None
    generator.sigma = None
    generator.c = init_c(train_dataset_loader, generator)
    generator.sigma = init_sigma(train_dataset_loader, generator)
    print("gsvdd_sigma: {}".format(generator.sigma))

    train_data = iter(train_dataset_loader)
    process = tqdm(range(START_ITER, END_ITER), desc='{AUC: }')

    for iteration in process:
        poly_lr_scheduler(optimizer_d, init_lr=LR, iter=iteration, max_iter=END_ITER)
        poly_lr_scheduler(optimizer_g, init_lr=LR, iter=iteration, max_iter=END_ITER)
        # --------------------- Loader ------------------------
        batch = next(train_data, None)
        if batch is None:
            # train_dataset_loader = load_train(train_path)
            train_data = iter(train_dataset_loader)
            batch = train_data.next()
        batch = batch[0]  # batch[1] contains labels
        batch_data = batch.to(device)
        data_tmp = batch_data.unfold(2, 32, p_stride).unfold(3, 32, p_stride)
        data_tmp = data_tmp.permute(0, 2, 3, 1, 4, 5)
        real_data = data_tmp.reshape(-1, 3, 32, 32)

        # --------------------- TRAIN E ------------------------
        optimizer_g.zero_grad()
        latent_z = generator.encoder(real_data)
        fake_data = generator(real_data)

        b,_ = latent_z.shape
        # Reconstruction loss
        weight = 0.85
        ms_ssim_batch_wise = 1 - ms_ssim(real_data, fake_data, data_range=data_range,
                                         size_average=True, win_size=3, weights=ssim_weights)
        l1_batch_wise = l1_criterion(real_data, fake_data)/data_range
        ms_ssim_l1 = weight * ms_ssim_batch_wise + (1 - weight) * l1_batch_wise

        ############ Interplote ############
        e1 = torch.flip(latent_z, dims=[0])
        alpha = torch.FloatTensor(b, 1).uniform_(0, 0.5).to(device)
        e2 = alpha * latent_z + (1 - alpha) * e1
        g2 = generator.generate(e2)
        reg_inter = torch.mean(discriminator(g2) ** 2)

        ############ DSVDD ############
        diff = (latent_z - generator.c) ** 2
        dist = -1 * (torch.sum(diff, dim=1) / generator.sigma)
        svdd_loss = torch.mean(1 - torch.exp(dist))

        encoder_loss = ms_ssim_l1 + svdd_loss + 0.1 * reg_inter

        encoder_loss.backward()
        optimizer_g.step()

        ############ Discriminator ############
        optimizer_d.zero_grad()
        g2 = generator.generate(e2).detach()
        fake_data = generator(real_data).detach()
        d_loss_front = torch.mean((discriminator(g2) - alpha) ** 2)
        gamma = 0.2
        tmp = fake_data + gamma * (real_data - fake_data)
        d_loss_back = torch.mean(discriminator(tmp) ** 2)
        d_loss = d_loss_front + d_loss_back
        d_loss.backward()
        optimizer_d.step()

        if iteration % int((train_size / BATCH_SIZE) * 10) == 0 and iteration != 0:
            generator.sigma = init_sigma(train_dataset_loader, generator)
            generator.c = init_c(train_dataset_loader, generator)

        if recorder is not None:
            recorder.record(loss=svdd_loss, epoch=int(iteration / BATCH_SIZE),
                            num_batches=len(train_data), n_batch=iteration, loss_name='DSVDD')

            recorder.record(loss=torch.mean(dist), epoch=int(iteration / BATCH_SIZE),
                            num_batches=len(train_data), n_batch=iteration, loss_name='DIST')

            recorder.record(loss=ms_ssim_batch_wise, epoch=int(iteration / BATCH_SIZE),
                            num_batches=len(train_data), n_batch=iteration, loss_name='MS-SSIM')

            recorder.record(loss=l1_batch_wise, epoch=int(iteration / BATCH_SIZE),
                            num_batches=len(train_data), n_batch=iteration, loss_name='L1')

        if iteration % int((train_size / BATCH_SIZE) * 10) == 0 or iteration == END_ITER - 1:
            is_end = True if iteration == END_ITER-1 else False
            test_auc, AUC_LIST = validation(NORMAL_NUM, iteration, generator, discriminator, real_data, fake_data, is_end, AUC_LIST, END_ITER)
            process.set_description("{AUC: %.5f}" % test_auc)
            torch.save(optimizer_g.state_dict(), ckpt_path + '/optimizer/g_opt.pth')
            torch.save(optimizer_d.state_dict(), ckpt_path + '/optimizer/d_opt.pth')


def validation(NORMAL_NUM, iteration, generator, discriminator, real_data, fake_data, is_end, AUC_LIST, END_ITER):
    generator.eval()
    discriminator.eval()
    # resnet.eval()
    y     = []
    score = []
    normal_gsvdd = []
    abnormal_gsvdd = []
    normal_recon = []
    abnormal_recon = []
    test_root = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/test/'.format(NORMAL_NUM)
    list_test = os.listdir(test_root)

    with torch.no_grad():
        for i in range(len(list_test)):
            current_defect = list_test[i]
            # print(current_defect)
            test_path = test_root + "{}".format(current_defect)
            valid_dataset_loader = load_test(test_path, sample_rate=1.)

            for index, (images, label) in enumerate(valid_dataset_loader):
                # img = five_crop_ready(images)
                img_tmp = images.to(device)
                img_tmp = img_tmp.unfold(2, 32, p_stride).unfold(3, 32, p_stride)
                # img     = extract_patch(img_tmp)
                img_tmp = img_tmp.permute(0, 2, 3, 1, 4, 5)
                img = img_tmp.reshape(-1, 3, 32, 32)

                latent_z = generator.encoder(img)
                generate_result = generator(img)

                weight = 0.85

                ms_ssim_batch_wise = 1 - ms_ssim(img, generate_result, data_range=data_range,
                                                 size_average=False, win_size=3, weights=ssim_weights)
                l1_batch_wise = (img - generate_result) / data_range
                l1_batch_wise = l1_batch_wise.mean(1).mean(1).mean(1)
                ms_ssim_l1 = weight * ms_ssim_batch_wise + (1 - weight) * l1_batch_wise

                diff = (latent_z - generator.c) ** 2
                dist = -1 * torch.sum(diff, dim=1) / generator.sigma
                guass_svdd_loss = 1 - torch.exp(dist)

                anormaly_score = ((0.5 * ms_ssim_l1 + 0.5 * guass_svdd_loss).max()).cpu().detach().numpy()
                score.append(float(anormaly_score))


                if label[0] == "good":
                    # if is_end:
                        # normal_gsvdd.append(float(guass_svdd_loss.max().cpu().detach().numpy()))
                        # normal_recon.append(float(ms_ssim_l1.max().cpu().detach().numpy()))
                    y.append(0)
                else:
                    # if is_end:
                        # abnormal_gsvdd.append(float(guass_svdd_loss.max().cpu().detach().numpy()))
                        # abnormal_recon.append(float(ms_ssim_l1.max().cpu().detach().numpy()))
                    y.append(1)

            ###################################################
    # if is_end:
    #     Helper.plot_2d_chart(x1=numpy.arange(0, len(normal_gsvdd)), y1=normal_gsvdd, label1='normal_loss',
    #                          x2=numpy.arange(len(normal_gsvdd), len(normal_gsvdd) + len(abnormal_gsvdd)),
    #                          y2=abnormal_gsvdd, label2='abnormal_loss',
    #                          title="{}: {}".format(NORMAL_NUM, "gsvdd loss"),
    #                          save_path="./plot/{}_gsvdd".format(NORMAL_NUM))
    # # if True:
    #     Helper.plot_2d_chart(x1=numpy.arange(0, len(normal_recon)), y1=normal_recon, label1='normal_loss',
    #                          x2=numpy.arange(len(normal_recon), len(normal_recon) + len(abnormal_recon)),
    #                          y2=abnormal_recon, label2='abnormals_loss',
    #                          title="{}: {}".format(NORMAL_NUM, "recon loss"),
    #                          save_path="./plot/{}_gsvdd_{}".format(NORMAL_NUM, iteration))
    fpr, tpr, thresholds = metrics.roc_curve(y, score, pos_label=1)
    auc_result = auc(fpr, tpr)
    AUC_LIST.append(auc_result)
    # tqdm.write(str(auc_result), end='.....................')
    auc_file = open(ckpt_path + "/auc.txt", "a")
    auc_file.write('Iter {}:            {}\r\n'.format(str(iteration), str(auc_result)))
    auc_file.close()
    if iteration == END_ITER - 1:
        auc_file = open(ckpt_path + "/auc.txt", "a")
        auc_file.write('BEST AUC -> {}\r\n'.format(max(AUC_LIST)))
        auc_file.close()
    return auc_result, AUC_LIST


if __name__ == "__main__":
    args = parser.parse_args()
    NORMAL_NUM_LIST = set(args.num)
    # sample_rate = args.sample_rate
    sub_category = {key: category[key] for key in args.num}
    print(sub_category)
    # exit(0)
    print("SAMPLE RATE: {}".format(args.sample_rate))
    for key in sorted(sub_category):
        NORMAL_NUM = sub_category[key]
        # NORMAL_NUM = category[key] if key.isdigit() else key
        print('Current Item: {}'.format(NORMAL_NUM))

        train_path = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/train/'.format(NORMAL_NUM)
        test_root = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/test/'.format(NORMAL_NUM)
        gt_root = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/ground_truth/'.format(NORMAL_NUM)

        # current_ckpt = "23999"
        Experiment_name = 'No.{}_p32_IGD'.format(str(NORMAL_NUM), sig_f, rec_f, svd_f)
        # Experiment_name = 'No.{}_vae'.format(str(NORMAL_NUM))

        recorder = Recorder(Experiment_name, 'MVTec_No.{}'.format(str(NORMAL_NUM)))

        save_root_path = './p32/check_points'
        
        create_dir(save_root_path)
        create_dir(os.path.join(save_root_path, "p32_SR-{}".format(args.sample_rate)))
        ckpt_path = './p32/check_points/p32_SR-{}/{}'.format(args.sample_rate, Experiment_name)

        if not os.path.exists(ckpt_path):
            os.mkdir(path=ckpt_path)
        auc_file = open(ckpt_path + "/auc.txt", "w")
        auc_file.close()

        generator     = twoin1Generator(64, latent_dimension=latent_dimension)
        discriminator = VisualDiscriminator(64)

        path = './Encoder_KD_ckpt'
        generator.pretrain.load_state_dict(torch.load(path))
        for param in generator.pretrain.parameters():
            param.requires_grad = False

        optimizer_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0, 0.9), weight_decay=1e-6)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0, 0.9))
        train(args, NORMAL_NUM, generator, discriminator, optimizer_g, optimizer_d)

