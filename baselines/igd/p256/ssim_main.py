import os
import numpy
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import autograd
from torch import optim
import torch.nn.init as init
from timeit import default_timer as timer
import cv2
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from pytorch_msssim import ms_ssim, ssim
from tqdm import tqdm
import numpy as np
# from ... import Helper
# from ... import Recorder
import Helper
from Recorder import Recorder
from p256.ssim_module import *
from p256.mvtec_module import twoin1Generator256, VisualDiscriminator256, Encoder_256
from p256.mvtex_data_loader import *
from torch.autograd import Variable
import argparse
device = torch.device("cuda:0")
print(">> Device Info: {} is in use".format(device))
parser = argparse.ArgumentParser(description='CIFAR10 Training')
parser.add_argument('-n', '--num', nargs='+', type=int, help='<Required> Set flag', required=True)
parser.add_argument('-sr', '--sample_rate', default=1, type=float)

DIM = 32  # Model dimensionality
CRITIC_ITERS = 5  # How many iterations to train the critic for
GENER_ITERS = 1
N_GPUS = 1  # Number of GPUs
BATCH_SIZE = 16  # Batch size. Must be a multiple of N_GPUS

LAMBDA = 10  # Gradient pena1lty lambda hyperparameter
OUTPUT_DIM = 32 * 32 * 3  # Number of pixels in each image

MAX_EPOCH = 256
num_worker = 6
############################ Parameters ############################
latent_dimension = 128

msssim_weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

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

# category = {"pill"}
data_range = 2.1179 + 2.6400
####################################################################

USE_SSIM = True

LR = 1e-4  # 0.0001


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
                                                    drop_last=False)
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
        raise Exception("ERROR")
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def init_c(DataLoader, net, eps=0.1):
    generator.c = None
    c = torch.zeros((1, latent_dimension)).to(device)
    net.eval()
    n_samples = 0
    with torch.no_grad():
        for index, (images, label) in enumerate(DataLoader):
            # get the inputs of the batch
            img = images.to(device)
            outputs = net.encoder(img)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c


sig_f = 1
def init_sigma(DataLoader, net):
    generator.sigma = None
    net.eval()
    tmp_sigma = torch.tensor(0.0, dtype=torch.float).to(device)
    n_samples = 0
    with torch.no_grad():
        for index, (images, label) in enumerate(DataLoader):
            img = images.to(device)
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


def train(args, NORMAL_NUM,
          generator, discriminator,
          optimizer_g, optimizer_d):
    AUC_LIST = []

    global test_auc
    test_auc = 0
    generator.c = None
    generator.sigma = None

    train_path = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/train/good'.format(NORMAL_NUM)
    START_ITER = 0
    # train_size = len(os.listdir(train_path))

    BEST_AUC = 0
    generator.train()
    discriminator.train()

    train_dataset_loader, train_size = load_train(train_path, args.sample_rate)

    END_ITER = int((train_size / BATCH_SIZE) * MAX_EPOCH)

    generator.c = init_c(train_dataset_loader, generator)
    generator.c.requires_grad = False
    generator.sigma = init_sigma(train_dataset_loader, generator)
    generator.sigma.requires_grad = False

    print(generator.sigma)
    train_data = iter(train_dataset_loader)
    process = tqdm(range(START_ITER, END_ITER), desc='{AUC: }')

    for iteration in process:
        poly_lr_scheduler(optimizer_d, init_lr=LR, iter=iteration, max_iter=END_ITER)
        poly_lr_scheduler(optimizer_g, init_lr=LR, iter=iteration, max_iter=END_ITER)

        # --------------------- Loader ------------------------
        batch = next(train_data, None)
        if batch is None:
            # train_dataset_loader, _ = load_train(train_path, sample_rate=args.sample_rate)
            train_data = iter(train_dataset_loader)
            batch = train_data.next()
        batch = batch[0]  # batch[1] contains labels
        real_data = batch.to(device)

        # --------------------- TRAIN E ------------------------
        optimizer_g.zero_grad()
        b, c, _, _ = real_data.shape 
        latent_z = generator.encoder(real_data)
        fake_data = generator.generate(latent_z)

        # Reconstruction loss
        weight = 0.85
        ms_ssim_batch_wise = 1 - ms_ssim(real_data, fake_data, data_range=data_range,
                                         size_average=True, win_size=11, weights=msssim_weight)
        l1_batch_wise = l1_criterion(real_data, fake_data) / data_range
        ms_ssim_l1 = weight * ms_ssim_batch_wise + (1 - weight) * l1_batch_wise

        ############ Interplote ############
        e1 = torch.flip(latent_z, dims=[0])
        alpha = torch.FloatTensor(b, 1).uniform_(0, 0.5).to(device)
        e2 = alpha * latent_z + (1 - alpha) * e1
        g2 = generator.generate(e2)
        reg_inter = torch.mean(discriminator(g2) ** 2)

        ############ GAC ############
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

        # ------------------ RECORDER ------------------
        if recorder is not None:
            recorder.record(loss=svdd_loss, epoch=int(iteration / BATCH_SIZE),
                            num_batches=len(train_data), n_batch=iteration, loss_name='GAC')

            recorder.record(loss=torch.mean(dist), epoch=int(iteration / BATCH_SIZE),
                            num_batches=len(train_data), n_batch=iteration, loss_name='DIST')

            recorder.record(loss=ms_ssim_batch_wise, epoch=int(iteration / BATCH_SIZE),
                            num_batches=len(train_data), n_batch=iteration, loss_name='MS-SSIM')

            recorder.record(loss=l1_batch_wise, epoch=int(iteration / BATCH_SIZE),
                            num_batches=len(train_data), n_batch=iteration, loss_name='L1')

        if iteration % int((train_size / BATCH_SIZE) * 5) == 0 or iteration == END_ITER - 1:
            is_end = True if iteration == END_ITER - 1 else False
            test_auc, AUC_LIST = validation(NORMAL_NUM, iteration, generator, discriminator, real_data, fake_data, is_end, AUC_LIST, END_ITER)
            process.set_description("{AUC: %.5f}" % test_auc)
            opt_path = ckpt_path + '/optimizer'
            if not os.path.exists(opt_path):
                os.mkdir(path=opt_path)
            torch.save(optimizer_g.state_dict(), ckpt_path + '/optimizer/g_opt.pth')
            torch.save(optimizer_d.state_dict(), ckpt_path + '/optimizer/d_opt.pth')



def validation(NORMAL_NUM, iteration, generator, discriminator, real_data, fake_data, is_end, AUC_LIST, END_ITER):
    discriminator.eval()
    generator.eval()
    # resnet.eval()
    y = []
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
                img = images.to(device)
                # latent_z = generator.encoder(img)
                latent_z = generator.encoder(img)
                generate_result = generator(img)

                ############################## Normal #####################

                for visual_index in range(latent_z.shape[0]):
                    weight = 0.85
                    tmp_org_unsq = img[visual_index].unsqueeze(0)
                    tmp_rec_unsq = generate_result[visual_index].unsqueeze(0)

                    ms_ssim_batch_wise = 1 - ms_ssim(tmp_org_unsq, tmp_rec_unsq, data_range=data_range,
                                                     size_average=True, win_size=11, weights=msssim_weight)
                    l1_loss = l1_criterion(img[visual_index], generate_result[visual_index]) / data_range
                    ms_ssim_l1 = weight * ms_ssim_batch_wise + (1 - weight) * l1_loss

                    diff = (latent_z[visual_index] - generator.c) ** 2
                    dist = -1 * torch.sum(diff, dim=1) / generator.sigma
                    guass_svdd_loss = 1 - torch.exp(dist)

                    anormaly_score = (0.5 * ms_ssim_l1 + 0.5 * guass_svdd_loss).cpu().detach().numpy()
                    score.append(float(anormaly_score))

                    la = label[visual_index]  # .cpu().detach().numpy()

                    if la == "good":
                        y.append(0)
                    else:
                        y.append(1)
            ###################################################

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

        # Download MVTEC_AD from "ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz" 
        train_path = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/train/'.format(NORMAL_NUM)
        test_root = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/test/'.format(NORMAL_NUM)
        gt_root = '/home/user/Documents/Public_Dataset/MVTec_AD/{}/ground_truth/'.format(NORMAL_NUM)

        Experiment_name = 'No.{}_p256_IGD'.format(str(NORMAL_NUM))
        
        recorder = Recorder(Experiment_name, 'MVTec_No.{}'.format(str(NORMAL_NUM)))

        save_root_path = './p256/check_points'
        create_dir(save_root_path)
        save_root_path = os.path.join(save_root_path, "IGD_wo_inter")
        create_dir(save_root_path)
        ckpt_path = os.path.join(save_root_path, "p256_SR-{}".format(args.sample_rate))
        create_dir(ckpt_path)
        ckpt_path = os.path.join(ckpt_path, Experiment_name)

        if not os.path.exists(ckpt_path):
            os.mkdir(path=ckpt_path)
        auc_file = open(ckpt_path + "/auc.txt", "w")
        auc_file.close()

        generator = twoin1Generator256(64, latent_dimension=latent_dimension)
        discriminator = VisualDiscriminator256(64)

        for param in generator.pretrain.parameters():
            param.requires_grad = False

        generator.to(device)
        discriminator.to(device)
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0, 0.9), weight_decay=1e-6)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0, 0.9))

        train(args, NORMAL_NUM, generator, discriminator, optimizer_g, optimizer_d)
