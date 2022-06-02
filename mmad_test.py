import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from models.mmad.reconstruction_network.depth import DepthRecons
from models.mmad.reconstruction_network.rgb import RGBRecons 
from models.mmad.seg_network.depth import DepthSeg
from models.mmad.seg_network.rgb import RGBSeg
from configuration.mmad.config import parse_arguments_mmad 
from tools.utilize import * 
from data_io.mvtec3d import MVTec3D, MVTecCL3D, mvtec3d_classes
import yaml
import numpy as np
from metrics.auc_precision_recall import get_auroc, get_precision_recall, get_ap

if __name__ == '__main__':
    args = parse_arguments_mmad() 

    args = parse_arguments_mmad() 

    with open('./configuration/mmad/{}.yaml'.format(args.dataset), 'r') as f:
        para_dict = yaml.load(f, Loader=yaml.SafeLoader)
    
    para_dict = merge_config(para_dict, args)
    print(para_dict)

    seed_everything(para_dict['seed'])

    device, device_ids = parse_device_list(para_dict['gpu_ids'], int(para_dict['gpu_id']))
    device = torch.device("cuda", device)

    mvtec3d_transform = {'data_size':para_dict['data_size'],
                         'mask_size':para_dict['mask_size']}
    
    if para_dict['class_names'] == 'all':
        class_names = mvtec3d_classes() 
    else: 
        class_names = para_dict['class_names']

    rgb_ck_path = os.path.join(para_dict['ck_path'], 'rgb') 
    depth_ck_path = os.path.join(para_dict['ck_path'], 'depth')

    rgb_recons_ck_path = os.path.join(rgb_ck_path, 'recons')
    rgb_seg_ck_path = os.path.join(rgb_ck_path, 'seg')

    depth_recons_ck_path = os.path.join(depth_ck_path, 'recons')
    depth_seg_ck_path = os.path.join(depth_ck_path, 'seg')

    rgb_obj_ap_pixel_list = []
    rgb_obj_auroc_pixel_list = []
    rgb_obj_ap_image_list = []
    rgb_obj_auroc_image_list = []

    depth_obj_ap_pixel_list = []
    depth_obj_auroc_pixel_list = []
    depth_obj_ap_image_list = []
    depth_obj_auroc_image_list = []

    img_dim = mvtec3d_transform['data_size']

    for cls in class_names:
        rgb_recons = RGBRecons(inc=3, fin_ouc=3).to(device)
        rgb_seg = RGBSeg(inc=3, fin_ouc=2).to(device)

        depth_recons = DepthRecons(inc=3, fin_ouc=3).to(device)
        depth_seg = DepthSeg(inc=3, fin_ouc=3).to(device)

        load_model(model=rgb_recons, file_path=rgb_recons_ck_path, description=cls+str(para_dict['num_epochs'])) 
        load_model(model=rgb_seg, file_path=rgb_seg_ck_path, description=cls+str(para_dict['num_epochs'])) 

        load_model(model=depth_recons, file_path=depth_recons_ck_path, description=cls+str(para_dict['num_epochs'])) 
        load_model(model=depth_seg, file_path=depth_seg_ck_path, description=cls+str(para_dict['num_epochs'])) 

        rgb_recons.eval()
        rgb_seg.eval()

        depth_recons.eval()
        depth_seg.eval() 

        if para_dict['dataset'] == 'mvtec3d':
            if para_dict['cl']:
                valid_dataset = MVTecCL3D()
            else: 
                valid_dataset = MVTec3D(data_path=para_dict['data_path'], class_names=cls,
                                        phase='test', depth_duplicate=para_dict['depth_duplicate'],
                                        data_transform=mvtec3d_transform) 

        valid_loader = DataLoader(valid_dataset, num_workers=para_dict['num_workers'],
                                  batch_size=para_dict['batch_size'], shuffle=False) 

        total_rgb_pixel_scores = np.zeros(img_dim * img_dim * len(valid_dataset))
        total_rgb_gt_pixel_scores = np.zeros(img_dim * img_dim * len(valid_dataset))

        total_depth_pixel_scores = np.zeros(img_dim * img_dim * len(valid_dataset))
        total_depth_gt_pixel_scores = np.zeros(img_dim * img_dim * len(valid_dataset))
        
        mask_cnt = 0

        rgb_anomaly_score_gt = []
        rgb_anomaly_score_prediction = []

        depth_anomaly_score_gt = []
        depth_anomaly_score_prediction = []
        
        
        for idx, batch in enumerate(valid_loader):
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            mask = batch['mask'].to(device)
            label = batch['label'].to(device)

            rgb_anomaly_score_gt.append(label)
            depth_anomaly_score_gt.append(label)

            # mask: convert torch format into open-cv format 
            mask_cv = mask.detach().numpy()[0. :, :, :].transpose((1, 2, 0))

            if para_dict['backbone_model'] == 'DRAEM':

                rgb_hat = RGBRecons(rgb)
                rgb_joined = torch.cat((rgb, rgb_hat), dim=1)
                rgb_out_mask = RGBSeg(rgb_joined)
                rgb_out_mask_softmax = torch.softmax(rgb_out_mask, dim=1)
                
                depth_hat = DepthRecons(depth) 
                depth_joined = torch.cat((depth, depth_hat), dim=1)
                depth_out_mask = DepthSeg(depth_joined)
                depth_out_mask_softmax = torch.softmax(depth_out_mask, dim=1)

                # segmentation output mask size  
                rgb_out_mask_cv = rgb_out_mask_softmax[0, 1, :, :].detach().cpu().numpy()
                depth_out_mask_cv = depth_out_mask_softmax[0, 1, :, :].detach().cpu().numpy()
                
                # smoothing mask and convert it into nympy format 
                rgb_out_mask_averaged = torch.nn.functional_avg_pool2d(input=rgb_out_mask_softmax,
                                                                       kernel_size=para_dict['smooth_kernel_size'],
                                                                       stride=1,
                                                                       padding=para_dict['smooth_kernel_size']//2).cpu().detach().numpy

                depth_out_mask_averaged = torch.nn.functional_avg_pool2d(input=depth_out_mask_softmax,
                                                                         kernel_size=para_dict['smooth_kernel_size'],
                                                                         stride=1,
                                                                         padding=para_dict['smooth_kernel_size']//2).cpu().detach().numpy
                
                rgb_score = np.max(rgb_out_mask_averaged)
                depth_score = np.max(depth_out_mask_averaged)

                rgb_anomaly_score_prediction.append(rgb_score)
                depth_anomaly_score_prediction.append(depth_score)

                flat_rgb_true_mask = mask_cv.flatten()                                            
                flat_rgb_out_mask = rgb_out_mask_cv.flatten()

                flat_depth_true_mask = mask_cv.flatten() 
                flat_depth_out_mask = depth_out_mask_cv.flatten()  

                total_rgb_pixel_scores[mask_cnt * img_dim * img_dim: (mask_cnt + 1) * img_dim * img_dim] = flat_rgb_out_mask
                total_rgb_gt_pixel_scores[mask_cnt * img_dim * img_dim: (mask_cnt + 1) * img_dim * img_dim] = flat_rgb_true_mask

                total_depth_pixel_scores[mask_cnt * img_dim * img_dim: (mask_cnt + 1) * img_dim * img_dim] = flat_depth_out_mask
                total_depth_gt_pixel_scores[mask_cnt * img_dim * img_dim: (mask_cnt + 1) * img_dim * img_dim] = flat_depth_true_mask

                mask_cnt += 1

        
        rgb_anomaly_score_prediction = np.array(rgb_anomaly_score_prediction)
        rgb_anomaly_score_gt = np.array(rgb_anomaly_score_gt)

        depth_anomaly_score_prediction = np.array(depth_anomaly_score_prediction)
        depth_anomaly_score_gt = np.array(depth_anomaly_score_gt)

        rgb_auroc = get_auroc(rgb_anomaly_score_gt, rgb_anomaly_score_prediction)
        depth_auroc = get_auroc(depth_anomaly_score_gt, depth_anomaly_score_prediction)

        rgb_ap = get_ap(rgb_anomaly_score_gt, rgb_anomaly_score_prediction)
        depth_ap = get_ap(depth_anomaly_score_gt, depth_anomaly_score_prediction)

        total_rgb_gt_pixel_scores = total_rgb_gt_pixel_scores.astype(np.uint8)
        total_rgb_gt_pixel_scores = total_rgb_pixel_scores[: img_dim * img_dim * mask_cnt]
        total_rgb_pixel_scores = total_rgb_pixel_scores[:img_dim * img_dim * mask_cnt]

        total_depth_gt_pixel_scores = total_depth_gt_pixel_scores.astype(np.uint8)
        total_depth_gt_pixel_scores = total_depth_pixel_scores[: img_dim * img_dim * mask_cnt]
        total_depth_pixel_scores = total_depth_pixel_scores[: img_dim * img_dim * mask_cnt]

        rgb_auroc_pixel = get_auroc(total_rgb_gt_pixel_scores, total_rgb_pixel_scores)
        depth_auroc_pixel = get_auroc(total_depth_gt_pixel_scores, total_depth_pixel_scores)

        rgb_ap_pixel = get_ap(total_rgb_gt_pixel_scores, total_rgb_pixel_scores)
        depth_ap_pixel = get_ap(total_depth_gt_pixel_scores, total_depth_pixel_scores)

        


                

                 
            