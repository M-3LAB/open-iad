from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple, Union
import os
from scipy.ndimage import gaussian_filter
from copy import deepcopy
import matplotlib.ticker as ticker
from arguments import get_args
from models import ResNetModel, ViT, Dis, Discat, PANDA, CutPaste, CSFlow, NetCSFlow, DRAEM, NetDRAEM, RevDis, NetRevDis, CLFlow, NetCLFlow
from datasets import get_mvtec_dataloaders, get_mtd_mvtec_dataloaders, get_joint_mtd_mvtec_dataloaders
from utils.optimizer import get_optimizer
from utils.density import GaussianDensityTorch


def preprocess_batch(data, device):
    '''move data to device and reshape image'''
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    inputs = inputs.view(-1, *inputs.shape[-3:])
    return inputs, labels

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None

# @staticmethod
def plot_tsne(labels, embeds, defect_name=None, save_path = None, **kwargs: Dict[str, Any]):
    """t-SNE visualize
    Args:
        labels (Tensor): labels of test and train
        embeds (Tensor): embeds of test and train
        defect_name ([str], optional): same as <defect_name> in roc_auc. Defaults to None.
        save_path ([str], optional): same as <defect_name> in roc_auc. Defaults to None.
        kwargs (Dict[str, Any]): hyper parameters of t-SNE which will change final result
            n_iter (int): > 250, default = 1000
            learning_rate (float): (10-1000), default = 100
            perplexity (float): (5-50), default = 28
            early_exaggeration (float): change it when not converging, default = 12
            angle (float): (0.2-0.8), default = 0.3
            init (str): "random" or "pca", default = "pca"
    """
    tsne = TSNE(
        n_components=2,
        verbose=1,
        n_iter=kwargs.get("n_iter", 1000),
        learning_rate=kwargs.get("learning_rate", 100),
        perplexity=kwargs.get("perplexity", 28),
        early_exaggeration=kwargs.get("early_exaggeration", 12),
        angle=kwargs.get("angle", 0.3),
        init=kwargs.get("init", "pca"),
    )
    embeds, labels = shuffle(embeds, labels)
    tsne_results = tsne.fit_transform(embeds)

    cmap = plt.cm.get_cmap("spring")
    colors = np.vstack((np.array([[0, 1. ,0, 1.]]), cmap([0, 256//3, (2*256)//3])))
    legends = ["good", "anomaly", "cutpaste", "cutpaste-scar"]
    (_, ax) = plt.subplots(1)
    plt.title(f't-SNE: {defect_name}')
    for label in torch.unique(labels):
        res = tsne_results[torch.where(labels==label)]
        ax.plot(*res.T, marker="*", linestyle="", ms=5, label=legends[label], color=colors[label])
        ax.legend(loc="best")
    plt.xticks([])
    plt.yticks([])

    save_images = save_path if save_path else './tnse_results1'
    os.makedirs(save_images, exist_ok=True)
    image_path = os.path.join(save_images, defect_name+'_tsne.jpg') if defect_name else os.path.join(save_images, 'tsne.jpg')
    plt.savefig(image_path)
    plt.close()
    return

def compare_histogram(scores, classes, start=0 ,thresh=2, interval=1, n_bins=64, name=None, save_path=None):
    classes = deepcopy(classes)
    classes[classes > 0] = 1
    scores[scores > thresh] = thresh
    bins = np.linspace(np.min(scores), np.max(scores), n_bins)
    scores_norm = scores[classes == 0]
    scores_ano = scores[classes == 1]

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=120)

    plt.hist(scores_norm, bins, alpha=0.5, density=True, label='non-defects', color='cyan', edgecolor="black")
    plt.hist(scores_ano, bins, alpha=0.5, density=True, label='defects', color='crimson', edgecolor="black")
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ticks = np.linspace(start, thresh, interval)
    labels = [str(i) for i in ticks[:-1]] + ['>' + str(thresh)]

    save_images = save_path if save_path else './his_results1'
    os.makedirs(save_images, exist_ok=True)
    image_path = os.path.join(save_images, name + '_his.jpg') if name else os.path.join(save_images, 'his.jpg')

    plt.xticks(ticks, labels=labels)
    plt.yticks(rotation=24)
    plt.xlabel(r'$-log(p(z))$', fontsize=22)
    plt.tick_params(labelsize=22)
    plt.rcParams.update({'font.size': 24})
    # plt.legend()
    # plt.grid(axis='y')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def get_dec_metrics(labels, predictions):
    metrics = {}
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['thresholds'] = thresholds
    f_measures = 2 * np.multiply(recall, precision) / (recall + precision + 1e-8)
    metrics['f_measures'] = f_measures
    ix_best = np.argmax(f_measures)
    metrics['ix_best'] = ix_best
    best_f_measure = f_measures[ix_best]
    metrics['best_f_measure'] = best_f_measure
    best_thr = thresholds[ix_best]
    metrics['best_thr'] = best_thr
    FPR, TPR, _ = roc_curve(labels, predictions)
    metrics['FPR'] = FPR
    metrics['TPR'] = TPR
    AUC = auc(FPR, TPR)
    metrics['AUC'] = AUC
    AP = auc(recall, precision)
    metrics['AP'] = AP
    decisions = predictions >= best_thr
    metrics['decisions'] = decisions
    FP, FN, TN, TP = calc_confusion_mat(decisions, labels)
    metrics['FP'] = FP
    metrics['FN'] = FN
    metrics['TN'] = TN
    metrics['TP'] = TP
    metrics['accuracy'] = (sum(TP) + sum(TN)) / (sum(TP) + sum(TN) + sum(FP) + sum(FN))
    return metrics

def calc_confusion_mat(D, Y):
    FP = (D != Y) & (Y.astype(np.bool) == False)
    FN = (D != Y) & (Y.astype(np.bool) == True)
    TN = (D == Y) & (Y.astype(np.bool) == False)
    TP = (D == Y) & (Y.astype(np.bool) == True)

    return FP, FN, TN, TP

def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count

def test(args):
    net = torch.load(f'checkpoints/{args.model.name}{args.seed}_{args.dataset.name}{args.dataset.data_incre_setting}_epochs{args.train.num_epochs}.pth')
    net.eval()
    net.to(args.device)
    density = torch.load(f'checkpoints/density_{args.model.name}{args.seed}_{args.dataset.name}{args.dataset.data_incre_setting}_epochs{args.train.num_epochs}.pth')
    # density.to(args.device)

    dataloaders_train, dataloaders_test, learned_tasks = [], [], []

    for t in range(args.dataset.n_tasks):
        if args.dataset.name == 'seq-mvtec':
            train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, data_train_nums = get_mvtec_dataloaders(
                args, t, dataloaders_train, dataloaders_test, learned_tasks)
        elif args.dataset.name == 'seq-mtd-mvtec':
            train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, data_train_nums = get_mtd_mvtec_dataloaders(
                args, t, dataloaders_train, dataloaders_test, learned_tasks)
        elif args.dataset.name == 'joint-mtd-mvtec':
            train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, data_train_nums = get_joint_mtd_mvtec_dataloaders(
                args, dataloaders_train, dataloaders_test, learned_tasks)

    for idx, (dataloader_test, learned_task) in enumerate(zip(dataloaders_test, learned_tasks)):
        labels, embeds = [], []
        with torch.no_grad():
            for x, label in dataloader_test:
                embed = net.forward_features(x.to(args.device))
                embeds.append(embed.cpu())
                labels.append(label.cpu())

        labels = torch.cat(labels)
        embeds = torch.cat(embeds)
        embeds = F.normalize(embeds, p=2, dim=1)

        distances = density.predict(embeds)
        roc_auc = roc_auc_score(labels, distances)
        # fpr, tpr, _ = roc_curve(labels, distances)
        # roc_auc1 = auc(fpr, tpr)
        print('data_type:', learned_task, 'auc:', roc_auc, '**' * 11)
        # True_lables = countX(labels, True)
        # False_lables = countX(labels, False)
        # metrics_dec = get_dec_metrics(np.array(labels).flatten(), np.array(distances).flatten())
        # FP, FN, TP, TN = list(
        #     map(sum, [metrics_dec["FP"], metrics_dec["FN"], metrics_dec["TP"], metrics_dec["TN"]]))
        # tpr = TP / (TP + FN)
        # fpr = FP / (FP + TN)
        # print('data_type:', learned_task, 'auc:', roc_auc, '**' * 11, FP, FN, TP, TN, True_lables, False_lables, tpr, fpr)


        if args.eval.visualization:
            name = f'{args.model.name}_task{len(learned_tasks)}_{learned_task}'
            tnse_save_path = f'./tnse_results1/{args.model.net}_{args.model.name}{args.seed}_{args.dataset.name}{args.dataset.data_incre_setting}_epochs{args.train.num_epochs}'
            his_save_path = f'./his_results1/{args.model.net}_{args.model.name}{args.seed}_{args.dataset.name}{args.dataset.data_incre_setting}_epochs{args.train.num_epochs}'
            plot_tsne(labels, np.array(embeds), defect_name=name,
                      save_path=tnse_save_path)
            if args.model.name == 'panda':
                start, thresh, interval = 0, 120, 1
            elif args.model.name == 'cutpaste' or args.model.name == 'upper':
                start, thresh, interval = 0, 100, 1
            elif args.model.name == 'dis' or args.model.name == 'discat':
                start, thresh, interval = 0, 160, 1
            compare_histogram(np.array(distances), labels, start=start,
                              thresh=thresh, interval=interval,
                              name=name, save_path=his_save_path)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    args = get_args()
    test(args)