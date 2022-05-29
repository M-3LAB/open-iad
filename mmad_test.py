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

    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []

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
        
        total_pixel_scores = np.zeros((mvtec3d_transform['data_size'] * mvtec3d_transform['data_size'] * len(valid_dataset)))
        total_gt_pixel_scores = np.zeros((mvtec3d_transform['data_size'] * mvtec3d_transform['data_size'] * len(valid_dataset)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []
        
        for idx, batch in enumerate(valid_loader):
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            mask = batch['mask'].to(device)
            label = batch['label'].to(device)
            