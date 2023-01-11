import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

from utils import *


def save_plot(subdir, name):
    exp_dir = os.path.join('viz', subdir, c.modelname)
    os.makedirs(exp_dir, exist_ok=True)
    plt.savefig(os.path.join(exp_dir, c.class_name + '_' + name), bbox_inches='tight', pad_inches=0)
    plt.close()


def viz_roc(y_score=None, y_test=None, name=''):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.clf()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for class ' + c.class_name)
    plt.legend(loc="lower right")
    plt.axis('equal')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    save_plot('roc', name)


def compare_histogram(scores, classes, thresh=None, n_bins=64, log=False, name=''):
    if log:
        scores = np.log(scores + 1e-8)

    if thresh is not None:
        if np.max(scores) < thresh:
            thresh = np.max(scores)
        scores[scores > thresh] = thresh
    bins = np.linspace(np.min(scores), np.max(scores), n_bins)
    scores_norm = scores[classes == 0]
    scores_ano = scores[classes == 1]

    plt.clf()
    plt.hist(scores_norm, bins, alpha=0.5, density=True, label='non-defects', color='cyan', edgecolor="black")
    plt.hist(scores_ano, bins, alpha=0.5, density=True, label='defects', color='crimson', edgecolor="black")

    ticks = np.linspace(np.min(scores), np.max(scores), 5)
    labels = ['{:.2f}'.format(i) for i in ticks[:-1]] + ['>' + '{:.2f}'.format(np.max(scores))]
    plt.xticks(ticks, labels=labels)
    plt.xlabel('Anomaly Score' if not log else 'Log Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y')
    save_plot('hist', name)


def viz_maps(img, depth, gt, map, fg, name='', norm=True):
    gt[fg == 0] = np.nan
    map = np.copy(map)
    map[fg == 0] = np.nan
    if norm:
        img = img.transpose((1, 2, 0))
        img *= np.array(c.norm_std)
        img += np.array(c.norm_mean)
    img = np.clip(img, 0, 1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs[0, 0].imshow(img)
    axs[0, 1].imshow(depth)
    axs[1, 0].imshow(gt, vmin=0, vmax=1)
    axs[1, 1].imshow(map)
    axs[0, 0].axis('off')
    axs[0, 1].axis('off')
    axs[1, 0].axis('off')
    axs[1, 1].axis('off')
    save_plot('maps', name)
