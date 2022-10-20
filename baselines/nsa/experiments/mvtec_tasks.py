from sklearn import metrics
import torch
import numpy as np
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from scipy import ndimage
import cv2
import bisect
from .plotting_utils import plot_row


# when full_size is True, gets 256x256 images from test_dat but only feeds 224x224 into model and pads with zeros
# this is so that we can plot/calculate results for the entire image although the model is trained on 224x224 crops
# for some classes
def test_real_anomalies(model, test_dat, device='cuda', batch_size=16, show=True, full_size=True): 
    model.eval()
    model = model.to(device)
    loader = DataLoader(test_dat, batch_size=batch_size, shuffle=False)
    preds = []
    sample_preds = []
    sample_labels = []
    pixel_labels = []
    inputs = []
    for data, labels, masks in tqdm(loader, desc='predict'):
        inputs.append(data.cpu())
        if full_size:
            data = T.CenterCrop(224)(data)
        data = data.to(device)
        with torch.no_grad():
            pred = model.forward(data)
        if isinstance(pred, tuple):
            pred, _ = pred
        sample_preds.append(torch.mean(pred, dim=(1,2,3)).cpu().numpy())
        if full_size:
            pred = T.Pad(16)(pred)
        preds.append(pred.cpu().numpy())
        sample_labels.append(labels.cpu().numpy())
        pixel_labels.append(masks.cpu().numpy())

    inputs = torch.cat(inputs)
    preds = np.concatenate(preds)
    sample_preds = np.concatenate(sample_preds)
    sample_labels = np.concatenate(sample_labels)
    pixel_labels = np.concatenate(pixel_labels)

    pixel_ap = metrics.average_precision_score(pixel_labels.flatten(), preds.flatten())
    pixel_auroc = metrics.roc_auc_score(pixel_labels.flatten(), preds.flatten())
    pixel_pro = pro_score(pixel_labels, preds)

    sample_ap = metrics.average_precision_score(sample_labels, sample_preds)
    sample_auroc = metrics.roc_auc_score(sample_labels, sample_preds)

    if show:
        print('pixel AP: {:.5f}, AUROC: {:.5f}, PRO-score: {:.5f}'.format(pixel_ap, pixel_auroc, pixel_pro))
        print('sample AP: {:.5f}, AUROC: {:.5f}'.format(sample_ap, sample_auroc))
    return sample_ap, sample_auroc, pixel_ap, pixel_auroc, pixel_pro 


def pro_score(pixel_labels, preds, max_fpr=0.3, max_components=25):
    assert pixel_labels.shape == preds.shape
    if len(pixel_labels.shape) == 4:  # has a class-channel
        assert pixel_labels.shape[1] == 1  # only implemented for single class
        pixel_labels = pixel_labels[:,0]
        preds = preds[:,0]
    
    # calculate FPR
    fpr, _, thresholds = metrics.roc_curve(pixel_labels.flatten(), preds.flatten())
    split = len(fpr[fpr < max_fpr]) 
    # last thresh has fpr >= max_fpr
    fpr = fpr[:(split+1)]
    thresholds = thresholds[:(split+1)]
    neg_thresholds = -thresholds
    preds[preds < thresholds[-1]] = 0
    
    # calculate per-component-overlap for each threshold and match to global thresholds
    pro = np.zeros_like(fpr)
    total_components = 0
    for j in range(len(pixel_labels)):
        num_labels, label_img = cv2.connectedComponents(np.uint8(pixel_labels[j]))
        if num_labels > max_components:
            raise ValueError("Invalid label map: too many components (" + str(num_labels) + ")") 
        if num_labels == 1:  # only background
            continue
        total_components += num_labels - 1
    
        y_score = preds[j].flatten()
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_score.size - 1]
        thresholds_j = y_score[threshold_idxs]
        for k in range(1, num_labels):
            y_true = np.uint8(label_img == k).flatten()
            y_true = y_true[desc_score_indices]
            tps = np.cumsum(y_true)[threshold_idxs] 
            tpr = tps / tps[-1]

            # match tprs to global thresholds so that we can calculate pro
            right = len(thresholds)
            for tpr_t, t in zip(tpr[::-1], thresholds_j[::-1]):  # iterate in ascending order
                if t < thresholds[-1]:  # remove too small thresholds
                    continue
                i = bisect.bisect_left(neg_thresholds, -t, hi=right)  # search for negated as thresholds desc
                pro[i : right] += tpr_t
                right = i
    pro /= total_components

    if fpr[-1] > max_fpr:  # interpolate last value
        pro[-1] = ((max_fpr - fpr[-2]) * pro[-1] + (fpr[-1] - max_fpr) * pro[-2]) / (fpr[-1] - fpr[-2])
        fpr[-1] = max_fpr

    return metrics.auc(fpr, pro) / max_fpr
