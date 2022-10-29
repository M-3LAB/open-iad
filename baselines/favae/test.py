import os
import argparse
import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from tqdm import tqdm
from skimage.segmentation import mark_boundaries
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from skimage import morphology
from scipy.ndimage import gaussian_filter
from models.VAE import VAE
from datasets.mvtec import MVTecDataset
from func import feature_extractor, denormalization

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
plt.switch_backend('agg')


def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--obj', type=str, default='.')
    parser.add_argument('--data_type', type=str, default='mvtec')
    parser.add_argument('--data_path', type=str, default='.')
    parser.add_argument('--checkpoint_dir', type=str, default='.')
    parser.add_argument("--grayscale", action='store_true', help='color or grayscale input image')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_resize', type=int, default=128)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    args.save_dir = './' + args.data_type + '/' + args.obj + '/vgg_feature' + '/seed_{}/'.format(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load model and dataset
    args.input_channel = 1 if args.grayscale else 3
    model = VAE(input_channel=args.input_channel, z_dim=100).to(device)
    checkpoint = torch.load(args.checkpoint_dir)
    model.load_state_dict(checkpoint['model'])
    teacher = models.vgg16(pretrained=True).to(device)
    for param in teacher.parameters():
        param.requires_grad = False

    img_size = args.crop_size if args.img_resize != args.crop_size else args.img_resize
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    test_dataset = MVTecDataset(args.data_path, class_name=args.obj, is_train=False, resize=img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    scores, test_imgs, recon_imgs, gt_list, gt_mask_list = test(model, teacher, test_loader)
    scores = np.asarray(scores)
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
    gt_mask = np.asarray(gt_mask_list)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

    plt.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (args.obj, per_pixel_rocauc))
    plt.legend(loc="lower right")
    save_dir = args.save_dir + '/' + f'seed_{args.seed}' + '/' + 'pictures_{:.4f}'.format(
        threshold)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, args.obj + '_roc_curve.png'), dpi=100)

    plot_fig(args, test_imgs, recon_imgs, scores, gt_mask_list, threshold, save_dir)


def test(model, teacher, test_loader):
    model.eval()
    teacher.eval()
    MSE_loss = nn.MSELoss(reduction='none')
    scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    recon_imgs = []
    for (data, label, mask) in tqdm(test_loader):
        test_imgs.extend(data.cpu().numpy())
        gt_list.extend(label.cpu().numpy())
        gt_mask_list.extend(mask.cpu().numpy())
        with torch.no_grad():
            data = data.to(device)
            z, output, _, _ = model(data)
            # get model's intermediate outputs
            s_activations, _ = feature_extractor(z, model.decode, target_layers=['10', '16', '22'])
            t_activations, _ = feature_extractor(data, teacher.features, target_layers=['7', '14', '21'])

            score = MSE_loss(output, data).sum(1, keepdim=True)
            for i in range(len(s_activations)):
                s_act = model.adapter[i](s_activations[-(i + 1)])
                mse_loss = MSE_loss(s_act, t_activations[i]).sum(1, keepdim=True)
                score += F.interpolate(mse_loss, size=data.size(2), mode='bilinear', align_corners=False)

        score = score.squeeze().cpu().numpy()
        for i in range(score.shape[0]):
            score[i] = gaussian_filter(score[i], sigma=4)
        scores.extend(score)
        recon_imgs.extend(output.cpu().numpy())
    return scores, test_imgs, recon_imgs, gt_list, gt_mask_list


def plot_fig(args, test_img, recon_imgs, scores, gts, threshold, save_dir):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        recon_img = recon_imgs[i]
        recon_img = denormalization(recon_img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 6, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(recon_img)
        ax_img[1].title.set_text('Reconst')
        ax_img[2].imshow(gt, cmap='gray')
        ax_img[2].title.set_text('GroundTruth')
        ax = ax_img[3].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[3].imshow(img, cmap='gray', interpolation='none')
        ax_img[3].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[3].title.set_text('Predicted heat map')
        ax_img[4].imshow(mask, cmap='gray')
        ax_img[4].title.set_text('Predicted mask')
        ax_img[5].imshow(vis_img)
        ax_img[5].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, args.obj + '_{}_png'.format(i)), dpi=100)
        plt.close()


if __name__ == '__main__':
    main()
