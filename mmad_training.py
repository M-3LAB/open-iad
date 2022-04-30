import torch
import torch.nn as nn
import yaml
from configuration.mmad.config import parse_arguments_mmad 
from tools.utilize import *
from data_io.mvtec3d import MVTec3D, MVTecCL3D, mvtec3d_classes
from torch.utils.data import DataLoader
#from torch.utils.data.sampler import SubsetRandomSampler

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

        #valid_loader = DataLoader(valid_dataset, num_workers=para_dict['num_workers'],
        #                          batch_size=para_dict['batch_size'], shuffle=False)

    #TODO: Model 

    for i, batch in enumerate(train_loader):
        x, y, mask, depth_map, xyz = batch
        #img = batch['rgb']
        #label = batch['label']
        #depth_map = batch['depth_map']

    #TODO: Self-Supervised Training 

