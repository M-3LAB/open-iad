from sklearn import metrics
import torch
import numpy as np
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from scipy import ndimage
from .plotting_utils import plot_row


# when full_size is True, gets 256x256 images from test_dat but only feeds 224x224 into model and pads with zeros
# this is so that we can plot results for the entire image although the model is trained on 224x224 crops
def test_real_anomalies(model, test_dat, device='cuda', batch_size=16, show=True, plots=True, full_size=False): 
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
        data = data.to(device)
        if full_size:
            data = T.CenterCrop(224)(data)
        with torch.no_grad():
            pred = model.forward(data)
        if isinstance(pred, tuple):
            pred, _ = pred
        if full_size:
            pred = T.Pad(16)(pred)
        sample_preds.append(torch.mean(pred, dim=(1,2,3)).cpu().numpy())
        preds.append(pred.cpu().numpy())
        sample_labels.append(labels.cpu().numpy())
        pixel_labels.append(masks.cpu().numpy())

    inputs = torch.cat(inputs)
    preds = np.concatenate(preds)
    sample_preds = np.concatenate(sample_preds)
    sample_labels = np.concatenate(sample_labels)
    pixel_labels = np.concatenate(pixel_labels)

    if plots:
        fig, ax = plt.subplots(1, 3, figsize=(20, 40), dpi=150)
        plot_row([inputs, torch.tensor(pixel_labels), torch.tensor(preds)], 
                ['input', 'ground truth', 'prediction'], ax, grid_cols=20)
    else:
        fig = None

    sample_ap = metrics.average_precision_score(sample_labels, sample_preds)
    sample_auroc = metrics.roc_auc_score(sample_labels, sample_preds)

    if show:
        print('sample AP: {:.5f}, AUROC: {:.5f}'.format(sample_ap, sample_auroc))
        plt.show()
    return sample_ap, sample_auroc, fig