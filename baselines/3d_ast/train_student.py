import numpy as np
import torch
from os.path import join
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import config as c
from model import *
from utils import *


def train(train_loader, test_loader):
    student = Model(nf=not c.asymmetric_student, channels_hidden=c.channels_hidden_student, n_blocks=c.n_st_blocks)
    student.to(c.device)

    teacher = Model()
    teacher.net.load_state_dict(torch.load(os.path.join(MODEL_DIR, c.modelname + '_' + c.class_name + '_teacher.pth')))
    teacher.eval()
    teacher.to(c.device)

    optimizer = torch.optim.Adam(student.net.parameters(), lr=c.lr, eps=1e-08, weight_decay=1e-5)

    max_st_obs = Score_Observer('AUROC  max over maps')
    mean_st_obs = Score_Observer('AUROC mean over maps')

    for epoch in range(c.meta_epochs):
        # train some epochs
        student.train()
        if c.verbose:
            print(F'\nTrain epoch {epoch}')
        for sub_epoch in range(c.sub_epochs):
            train_loss = list()
            for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                depth, fg, labels, image, features = data
                depth, fg, image, features = to_device([depth, fg, image, features])
                fg = dilation(fg, c.dilate_size) if c.dilate_mask else fg

                optimizer.zero_grad()
                img_in = features if c.pre_extracted else image
                fg_down = downsampling(fg, (c.map_len, c.map_len), bin=False)

                with torch.no_grad():
                    z_t, jac_t = teacher(img_in, depth)

                z, jac = student(img_in, depth)
                loss = get_st_loss(z_t, z, fg_down)
                loss.backward()
                optimizer.step()

                train_loss.append(t2np(loss))

            mean_train_loss = np.mean(train_loss)
            if c.verbose and sub_epoch % 4 == 0:  # and epoch == 0:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))

        # evaluate
        student.eval()
        if c.verbose:
            print('\nCompute loss and scores on test set:')
        test_loss = list()
        test_labels = list()
        mean_st = list()
        max_st = list()

        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
                depth, fg, labels, image, features = data
                depth, fg, image, features = to_device([depth, fg, image, features])
                fg = dilation(fg, c.dilate_size) if c.dilate_mask else fg

                img_in = features if c.pre_extracted else image
                fg_down = downsampling(fg, (c.map_len, c.map_len), bin=False)
                z_t, jac_t = teacher(img_in, depth)

                z, jac = student(img_in, depth)

                st_loss = get_st_loss(z_t, z, fg_down, per_sample=True)
                st_pixel = get_st_loss(z_t, z, fg_down, per_pixel=True)

                if c.eval_mask:
                    st_pixel = st_pixel * fg_down[:, 0]

                mean_st.append(t2np(st_loss))
                max_st.append(np.max(t2np(st_pixel), axis=(1, 2)))
                test_loss.append(st_loss.mean().item())
                test_labels.append(labels)

        mean_st = np.concatenate(mean_st)
        max_st = np.concatenate(max_st)
        test_loss = np.mean(np.array(test_loss))

        if c.verbose:
            print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, test_loss))

        test_labels = np.concatenate(test_labels)
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        mean_st_obs.update(roc_auc_score(is_anomaly, mean_st), epoch,
                           print_score=c.verbose or epoch == c.meta_epochs - 1)
        max_st_obs.update(roc_auc_score(is_anomaly, max_st), epoch, print_score=c.verbose or epoch == c.meta_epochs - 1)

    if c.save_model:
        save_weights(student, 'student')

    return mean_st_obs, max_st_obs


if __name__ == "__main__":
    train_dataset(train)
