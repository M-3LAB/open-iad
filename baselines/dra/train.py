import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os

from dataloaders.dataloader import initDataloader
from modeling.net import DRA
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from modeling.layers import build_criterion
import random

import matplotlib.pyplot as plt

WEIGHT_DIR = './weights'


class Trainer(object):

    def __init__(self, args):
        self.args = args
        # Define Dataloader
        kwargs = {'num_workers': args.workers}
        self.train_loader, self.test_loader= initDataloader.build(args, **kwargs)
        if self.args.total_heads == 4:
            temp_args = args
            temp_args.batch_size = self.args.nRef
            temp_args.nAnomaly = 0
            self.ref_loader, _ = initDataloader.build(temp_args, **kwargs)
            self.ref = iter(self.ref_loader)

        self.model = DRA(args, backbone=self.args.backbone)

        if self.args.pretrain_dir != None:
            self.model.load_state_dict(torch.load(self.args.pretrain_dir))
            print('Load pretrain weight from: ' + self.args.pretrain_dir)

        self.criterion = build_criterion(args.criterion)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def generate_target(self, target, eval=False):
        targets = list()
        if eval:
            targets.append(target==0)
            targets.append(target)
            targets.append(target)
            targets.append(target)
            return targets
        else:
            temp_t = target != 0
            targets.append(target == 0)
            targets.append(temp_t[target != 2])
            targets.append(temp_t[target != 1])
            targets.append(target != 0)
        return targets

    def training(self, epoch):
        train_loss = 0.0
        class_loss = list()
        for j in range(self.args.total_heads):
            class_loss.append(0.0)
        self.model.train()
        self.scheduler.step()
        tbar = tqdm(self.train_loader)
        for idx, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            if self.args.total_heads == 4:
                try:
                    ref_image = next(self.ref)['image']
                except StopIteration:
                    self.ref = iter(self.ref_loader)
                    ref_image = next(self.ref)['image']
                ref_image = ref_image.cuda()
                image = torch.cat([ref_image, image], dim=0)

            outputs = self.model(image, target)
            targets = self.generate_target(target)

            losses = list()
            for i in range(self.args.total_heads):
                if self.args.criterion == 'CE':
                    prob = F.softmax(outputs[i], dim=1)
                    losses.append(self.criterion(prob, targets[i].long()).view(-1, 1))
                else:
                    losses.append(self.criterion(outputs[i], targets[i].float()).view(-1, 1))

            loss = torch.cat(losses)
            loss = torch.sum(loss)

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            train_loss += loss.item()
            for i in range(self.args.total_heads):
                class_loss[i] += losses[i].item()

            tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / (idx + 1)))


    def normalization(self, data):
        return data

    def eval(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        class_pred = list()
        for i in range(self.args.total_heads):
            class_pred.append(np.array([]))
        total_target = np.array([])
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            if self.args.total_heads == 4:
                try:
                    ref_image = next(self.ref)['image']
                except StopIteration:
                    self.ref = iter(self.ref_loader)
                    ref_image = next(self.ref)['image']
                ref_image = ref_image.cuda()
                image = torch.cat([ref_image, image], dim=0)

            with torch.no_grad():
                outputs = self.model(image, target)
                targets = self.generate_target(target, eval=True)

                losses = list()
                for i in range(self.args.total_heads):
                    if self.args.criterion == 'CE':
                        prob = F.softmax(outputs[i], dim=1)
                        losses.append(self.criterion(prob, targets[i].long()))
                    else:
                        losses.append(self.criterion(outputs[i], targets[i].float()))

                loss = losses[0]
                for i in range(1, self.args.total_heads):
                    loss += losses[i]

            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

            for i in range(self.args.total_heads):
                if i == 0:
                    data = -1 * outputs[i].data.cpu().numpy()
                else:
                    data = outputs[i].data.cpu().numpy()
                class_pred[i] = np.append(class_pred[i], data)
            total_target = np.append(total_target, target.cpu().numpy())

        total_pred = self.normalization(class_pred[0])
        for i in range(1, self.args.total_heads):
            total_pred = total_pred + self.normalization(class_pred[i])

        with open(self.args.experiment_dir + '/result.txt', mode='a+', encoding="utf-8") as w:
            for label, score in zip(total_target, total_pred):
                w.write(str(label) + '   ' + str(score) + "\n")

        total_roc, total_pr = aucPerformance(total_pred, total_target)

        normal_mask = total_target == 0
        outlier_mask = total_target == 1
        plt.clf()
        plt.bar(np.arange(total_pred.size)[normal_mask], total_pred[normal_mask], color='green')
        plt.bar(np.arange(total_pred.size)[outlier_mask], total_pred[outlier_mask], color='red')
        plt.ylabel("Anomaly score")
        plt.savefig(args.experiment_dir + "/vis.png")
        return total_roc, total_pr

    def save_weights(self, filename):
        # if not os.path.exists(WEIGHT_DIR):
        #     os.makedirs(WEIGHT_DIR)
        torch.save(self.model.state_dict(), os.path.join(args.experiment_dir, filename))

    def load_weights(self, filename):
        path = os.path.join(WEIGHT_DIR, filename)
        self.model.load_state_dict(torch.load(path))

    def init_network_weights_from_pretraining(self):

        net_dict = self.model.state_dict()
        ae_net_dict = self.ae_model.state_dict()

        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        net_dict.update(ae_net_dict)
        self.model.load_state_dict(net_dict)

def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap;

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=48, help="batch size used in SGD")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=30, help="the number of epochs")
    parser.add_argument("--cont_rate", type=float, default=0.0, help="the outlier contamination rate in the training data")
    parser.add_argument("--test_threshold", type=int, default=0,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--test_rate", type=float, default=0.0,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--dataset", type=str, default='mvtecad', help="a list of data set names")
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--savename', type=str, default='model.pkl', help="save modeling")
    parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='./experiment/experiment_14', help="dataset root")
    parser.add_argument('--classname', type=str, default='capsule', help="dataset class")
    parser.add_argument('--img_size', type=int, default=448, help="dataset root")
    parser.add_argument("--nAnomaly", type=int, default=10, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='resnet18', help="backbone")
    parser.add_argument('--criterion', type=str, default='deviation', help="loss")
    parser.add_argument("--topk", type=float, default=0.1, help="topk in MIL")
    parser.add_argument('--know_class', type=str, default=None, help="set the know class for hard setting")
    parser.add_argument('--pretrain_dir', type=str, default=None, help="root of pretrain weight")
    parser.add_argument("--total_heads", type=int, default=4, help="number of head in training")
    parser.add_argument("--nRef", type=int, default=5, help="number of reference set")
    parser.add_argument('--outlier_root', type=str, default=None, help="OOD dataset root")
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    trainer = Trainer(args)


    argsDict = args.__dict__
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)
    with open(args.experiment_dir + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    print('Total Epoches:', trainer.args.epochs)
    trainer.model = trainer.model.to('cuda')
    trainer.criterion = trainer.criterion.to('cuda')
    for epoch in range(0, trainer.args.epochs):
        trainer.training(epoch)
    trainer.eval()
    trainer.save_weights(args.savename)

