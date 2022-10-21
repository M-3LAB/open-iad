import numpy as np
import torch
import torch.nn as nn

import argparse
import os

from dataloaders.dataloader import build_dataloader
from modeling.net import SemiADNet
from tqdm import tqdm
from utils import aucPerformance
from modeling.layers import build_criterion

class Trainer(object):

    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': args.workers}
        self.train_loader, self.test_loader = build_dataloader(args, **kwargs)

        self.model = SemiADNet(args)

        self.criterion = build_criterion(args.criterion)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        if args.cuda:
           self.model = self.model.cuda()
           self.criterion = self.criterion.cuda()

    def train(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            output = self.model(image)
            loss = self.criterion(output, target.unsqueeze(1).float())
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / (i + 1)))
        self.scheduler.step()

    def eval(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        total_pred = np.array([])
        total_target = np.array([])
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image.float())
            loss = self.criterion(output, target.unsqueeze(1).float())
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            total_pred = np.append(total_pred, output.data.cpu().numpy())
            total_target = np.append(total_target, target.cpu().numpy())
        roc, pr = aucPerformance(total_pred, total_target)
        return roc, pr

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(args.experiment_dir, filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=48, help="batch size used in SGD")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=50, help="the number of epochs")
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--weight_name', type=str, default='model.pkl', help="the name of model weight")
    parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='./experiment', help="experiment dir root")
    parser.add_argument('--classname', type=str, default='carpet', help="the subclass of the datasets")
    parser.add_argument('--img_size', type=int, default=448, help="the image size of input")
    parser.add_argument("--n_anomaly", type=int, default=10, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='resnet18', help="the backbone network")
    parser.add_argument('--criterion', type=str, default='deviation', help="the loss function")
    parser.add_argument("--topk", type=float, default=0.1, help="the k percentage of instances in the topk module")
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    trainer = Trainer(args)
    torch.manual_seed(args.ramdn_seed)

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    argsDict = args.__dict__
    with open(args.experiment_dir + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    for epoch in range(0, trainer.args.epochs):
        trainer.train(epoch)
    trainer.eval()
    trainer.save_weights(args.weight_name)

