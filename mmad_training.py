import torch
import torch.nn as nn
import yaml
from configuration.mmad.config import parse_arguments_mmad 
from tools.utilize import *
from data_io.mvtec3d import MVTec3D, MVTecCL3D, mvtec3d_classes
from torch.utils.data import DataLoader
#from torch.utils.data.sampler import SubsetRandomSampler
import os
from models.mmad.reconstruction_network.depth import DepthRecons
from models.mmad.reconstruction_network.rgb import RGBRecons 
from models.mmad.seg_network.depth import DepthSeg
from models.mmad.seg_network.rgb import RGBSeg
from tools.utilize import * 
from loss_function.ssim_loss import SSIMLoss 
from loss_function.focal_loss import FocalLoss

if __name__ == '__main__':

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

    if para_dict['dataset'] == 'mvtec3d':
        if para_dict['cl']:
            train_dataset = MVTecCL3D()
            valid_dataset = MVTecCL3D()
        else:
            train_dataset = MVTec3D(data_path=para_dict['data_path'], class_names=class_names,
                                    phase='train', depth_duplicate=para_dict['depth_duplicate'], 
                                    data_transform=mvtec3d_transform)

            valid_dataset = MVTec3D(data_path=para_dict['data_path'], class_names=class_names,
                                    phase='test', depth_duplicate=para_dict['depth_duplicate'],
                                    data_transform=mvtec3d_transform) 

    if not para_dict['cl']:
        train_loader = DataLoader(train_dataset,
                                  batch_size=para_dict['batch_size'],
                                  num_workers=para_dict['num_workers'],
                                  shuffle=False)

        valid_loader = DataLoader(valid_dataset, num_workers=para_dict['num_workers'],
                                  batch_size=para_dict['batch_size'], shuffle=False)

    rgb_ck_path = os.path.join(para_dict['ck_path'], 'rgb') 
    depth_ck_path = os.path.join(para_dict['ck_path'], 'depth')

    rgb_recons_ck_path = os.path.join(rgb_ck_path, 'recons')
    create_folders(rgb_recons_ck_path)

    rgb_seg_ck_path = os.path.join(rgb_ck_path, 'seg')
    create_folders(rgb_seg_ck_path)

    depth_recons_ck_path = os.path.join(depth_ck_path, 'recons')
    create_folders(depth_recons_ck_path)

    depth_seg_ck_path = os.path.join(depth_ck_path, 'seg')
    create_folders(depth_seg_ck_path)

    l2_loss = torch.nn.MSELoss().to(device)
    focal_loss = FocalLoss().to(device)
    ssim_loss = SSIMLoss().to(device)
    

    for cls in class_names: 
        rgb_recons = RGBRecons(inc=3, fin_ouc=3).to(device)
        depth_recons = DepthRecons(inc=1, fin_ouc=2).to(device)

        rgb_recons.apply(draem_weights_init)
        depth_recons.apply(draem_weights_init)
        
        rgb_seg = RGBSeg(inc=6, fin_ouc=2).to(device)
        depth_seg = DepthSeg(inc=2, fin_ouc=2).to(device)

        rgb_seg.apply(draem_weights_init)
        depth_seg.apply(draem_weights_init)

        rgb_optimizer = torch.optim.Adam([{"params": rgb_seg.parameters(), "lr": para_dict['lr']},
                                          {"params": rgb_recons.parameters(), "lr": para_dict['lr']}])

        depth_optimizer = torch.optim.Adam([{"params": depth_seg.parameters(), "lr": para_dict['lr']},
                                            {"params": depth_recons.parameters(), "lr": para_dict['lr']}])
        
        rgb_scheduler = torch.optim.lr_scheduler.MultiStepLR(rgb_optimizer, 
                                                             milestones=[para_dict['milestones_lower'], para_dict['milestones_higher']],
                                                             gamma=para_dict['gamma'])

        depth_scheduler = torch.optim.lr_scheduler.MultiStepLR(depth_optimizer, 
                                                               milestones=[para_dict['milestones_lower'], para_dict['milestones_higher']],
                                                               gamma=para_dict['gamma'])
        
        if para_dict['dataset'] == 'mvtec3d':
            if para_dict['cl']:
                train_dataset = MVTecCL3D()
                valid_dataset = MVTecCL3D()
            else:
                train_dataset = MVTec3D(data_path=para_dict['data_path'], class_names=cls,
                                        phase='train', depth_duplicate=para_dict['depth_duplicate'], 
                                        data_transform=mvtec3d_transform)

                valid_dataset = MVTec3D(data_path=para_dict['data_path'], class_names=cls,
                                        phase='test', depth_duplicate=para_dict['depth_duplicate'],
                                        data_transform=mvtec3d_transform) 

        if not para_dict['cl']:
            train_loader = DataLoader(train_dataset,
                                      batch_size=para_dict['batch_size'],
                                      num_workers=para_dict['num_workers'],
                                      shuffle=True)

            valid_loader = DataLoader(valid_dataset, num_workers=para_dict['num_workers'],
                                      batch_size=para_dict['batch_size'], shuffle=False) 
        
        for epoch in range(para_dict['num_epochs']):
            for _, batch in enumerate(train_loader): 
                if para_dict['aug_method'] == 'DRAEM':
                    rgb = batch['rgb'].to(device)
                    aug_rgb = batch['aug_rgb'].to(device)
                    depth = batch['depth'].to(device)
                    aug_depth = batch['aug_depth'].to(device)
                    aug_mask = batch['aug_mask'].to(device)

                    rgb_hat = rgb_recons(rgb)
                    rgb_joined = torch.cat((rgb_hat, aug_rgb), dim=1)

                    depth_hat = depth_recons(depth)
                    depth_joined = torch.cat((depth_hat, aug_depth), dim=1)

                    rgb_output_mask = rgb_seg(rgb_joined) 
                    rgb_output_mask_logit = torch.softmax(rgb_output_mask, dim=1)

                    depth_output_mask = depth_seg(depth_joined)
                    depth_output_mask_logit = torch.softmax(depth_output_mask, dim=1)

                    rgb_recons_loss = l2_loss(rgb, rgb_hat)                   
                    depth_recons_loss = l2_loss(depth, depth_hat)

                    rgb_ssim_loss = ssim_loss(rgb, rgb_hat) 
                    depth_ssim_loss = ssim_loss(depth, depth_hat)

                    rgb_seg_loss = focal_loss(rgb_output_mask_logit, aug_mask) 
                    depth_seg_loss = focal_loss(depth_output_mask_logit, aug_mask)

                    rgb_total_loss = (para_dict['lambda_recon'] * rgb_recons_loss + 
                                      para_dict['lambda_seg'] * rgb_seg_loss + 
                                      para_dict['lambda_ssim'] * rgb_ssim_loss)

                    depth_total_loss = (para_dict['lambda_recon'] * depth_recons_loss + 
                                        para_dict['lambda_seg'] * depth_seg_loss + 
                                        para_dict['lambda_ssim'] * depth_ssim_loss)
                    
                    total_loss = (para_dict['lambda_rgb'] * rgb_total_loss + 
                                  para_dict['lambda_depth'] * depth_total_loss)

                    rgb_optimizer.zero_grad()
                    depth_optimizer.zero_grad()

                    total_loss.backward()
                    
                    rgb_optimizer.step()
                    depth_optimizer.step()

            rgb_scheduler.step()
            depth_scheduler.step()

            #TODO: save rgb and depth model
            save_model()
            