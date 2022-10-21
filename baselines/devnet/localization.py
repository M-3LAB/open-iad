import numpy as np
import torch
from modeling.net import SemiADNet
from datasets import mvtecad
import cv2
import os
import argparse
from modeling.layers import build_criterion
from utils import aucPerformance
from scipy.ndimage.filters import gaussian_filter

np.seterr(divide='ignore',invalid='ignore')

def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

def farward_hook(module, input, output):
    fmap_block.append(output)

def convert_to_grayscale(im_as_arr):
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    if im_max > 0:
        grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def show_cam_on_image(img, mask, label, out_dir, name):

    img1 = img.copy()
    img[:, :, 0] = img1[:, :, 2]
    img[:, :, 1] = img1[:, :, 1]
    img[:, :, 2] = img1[:, :, 0]

    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.concatenate((img, cam), axis=1)
    cam = np.concatenate((cam, label), axis=1)

    path_cam_img = os.path.join(out_dir, args.classname + "_cam_" + name + ".jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--weight_name', type=str, default='model.pkl', help="the name of model weight")
    parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='./experiment', help="experiment dir root")
    parser.add_argument('--classname', type=str, default='carpet', help="the subclass of the datasets")
    parser.add_argument('--img_size', type=int, default=448, help="the image size of input")
    parser.add_argument("--n_anomaly", type=int, default=10, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='resnet18', help="the backbone network")
    parser.add_argument("--topk", type=float, default=0.1, help="the k percentage of instances in the topk module")
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.ramdn_seed)

    model = SemiADNet(args)
    model.load_state_dict(torch.load(os.path.join(args.experiment_dir, args.weight_name)))
    model = model.cuda()
    model.eval()

    fmap_block = list()
    grad_block = list()
    model.feature_extractor.net.layer4[1].conv2.register_forward_hook(farward_hook)
    model.feature_extractor.net.layer4[1].conv2.register_backward_hook(backward_hook)

    train_set = mvtecad.MVTecAD(args, train=False)

    outputs = list()
    fmaps = list()
    grads = list()
    seg_label = list()
    outliers_cam = list()
    input = list()
    outlier_scores = list()
    for i in train_set.outlier_idx:
        model.zero_grad()
        sample = train_set.getitem(i)
        inputs = sample['image'].view(1, 3, 448, 448).cuda()
        input.append(np.array(sample['raw_image']))
        inputs.requires_grad = True
        output = model(inputs)
        outlier_scores.append(output.data.cpu().numpy()[0][0])
        output.backward()

        grad = inputs.grad
        grad_temp = convert_to_grayscale(grad.cpu().numpy().squeeze(0))
        grad_temp = grad_temp.squeeze(0)
        grad_temp = gaussian_filter(grad_temp, sigma=4)
        outliers_cam.append(grad_temp)

        outputs.append(output.item())
        fmaps.append(fmap_block)
        grad_block.reverse()
        grads.append(grad_block)
        seg_label.append(np.array(sample['seg_label']))
        fmap_block = list()
        grad_block = list()

    for i, (cam, raw, label) in enumerate(zip(outliers_cam, input, seg_label)):
        raw = np.float32(cv2.resize(np.array(raw), (448, 448))) / 255
        label = cv2.resize(label, (448, 448)) / 255
        show_cam_on_image(raw, cam, label, os.path.join(args.experiment_dir, 'vis'), str(i))

    aucs = list()
    print("Detected anomaly: " + str(len(outliers_cam)))
    if len(outliers_cam) == 0:
        print('Cannot find anomaly image')
        exit()
    for cam, label in zip(outliers_cam, seg_label):
        label = cv2.resize(label, (448,448))
        label = label > 0
        cam_line = cam.reshape(-1)
        label_line = label[:,:,0].reshape(-1)
        auc, _ = aucPerformance(cam_line, label_line, prt=False)
        aucs.append(auc)
    aucs = np.array(aucs)
    print("Pixel-level AUC-ROC: %.4f" % (np.mean(aucs)))
    print('Visualization results are saved in: ' + os.path.join(args.experiment_dir, 'vis'))