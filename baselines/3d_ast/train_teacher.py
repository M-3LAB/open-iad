import numpy as np
import torch
from os.path import join
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import config as c
from model import *
from utils import *


def train(train_loader, test_loader):
    model = Model()
    model.to(c.device)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=c.lr, eps=1e-08, weight_decay=1e-5)

    mean_nll_obs = Score_Observer('AUROC mean over maps')
    max_nll_obs = Score_Observer('AUROC  max over maps')

    for epoch in range(c.meta_epochs):
        # train some epochs
        model.train()
        if c.verbose:
            print(F'\nTrain epoch {epoch}')
        for sub_epoch in range(c.sub_epochs):
            train_loss = list()
            for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                optimizer.zero_grad()

                depth, fg, labels, image, features = data
                depth, fg, labels, image, features = to_device([depth, fg, labels, image, features])
                fg = dilation(fg, c.dilate_size) if c.dilate_mask else fg

                img_in = features if c.pre_extracted else image
                fg_down = downsampling(fg, (c.map_len, c.map_len), bin=False)
                z, jac = model(img_in, depth)

                loss = get_nf_loss(z, jac, fg_down)
                train_loss.append(t2np(loss))

                loss.backward()
                optimizer.step()

            mean_train_loss = np.mean(train_loss)
            if c.verbose and sub_epoch % 4 == 0:  # and epoch == 0:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))

        # evaluate
        model.eval()
        if c.verbose:
            print('\nCompute loss and scores on test set:')
        test_loss = list()
        test_labels = list()
        img_nll = list()
        max_nlls = list()
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
                depth, fg, labels, image, features = data
                depth, fg, image, features = to_device([depth, fg, image, features])

                fg = dilation(fg, c.dilate_size) if c.dilate_mask else fg

                img_in = features if c.pre_extracted else image
                fg_down = downsampling(fg, (c.map_len, c.map_len), bin=False)
                z, jac = model(img_in, depth)
                loss = get_nf_loss(z, jac, fg_down, per_sample=True)
                nll = get_nf_loss(z, jac, fg_down, per_pixel=True)

                img_nll.append(t2np(loss))
                max_nlls.append(np.max(t2np(nll), axis=(-1, -2)))
                test_loss.append(loss.mean().item())
                test_labels.append(labels)

        img_nll = np.concatenate(img_nll)
        max_nlls = np.concatenate(max_nlls)
        test_loss = np.mean(np.array(test_loss))

        if c.verbose:
            print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, test_loss))

        test_labels = np.concatenate(test_labels)
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        mean_nll_obs.update(roc_auc_score(is_anomaly, img_nll), epoch,
                            print_score=c.verbose or epoch == c.meta_epochs - 1)
        max_nll_obs.update(roc_auc_score(is_anomaly, max_nlls), epoch,
                           print_score=c.verbose or epoch == c.meta_epochs - 1)

    if c.save_model:
        save_weights(model, 'teacher')

    return mean_nll_obs, max_nll_obs


if __name__ == "__main__":
    train_dataset(train)
