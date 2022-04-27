import torch
import torch.nn as nn
import yaml
from configuration.mmad.config import parse_arguments_mmad 
from tools.utilize import *
from data_io.mvtec3d import MVTec3D

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
    
    if para_dict['dataset'] == 'mvtec3d':

        train_dataset = MVTec3D(data_path=para_dict['data_path'],
                                         learning_mode=self.para_dict['learning_mode'],
                                         phase='train',
                                         data_transform=mvtec3d_transform,
                                         num_task=para_dict['num_task'])

        valid_dataset = MVTec3D(data_path=para_dict['data_path'],
                                learning_mode=para_dict['learning_mode'],
                                phase='test',
                                data_transform=mvtec3d_transform)


    #TODO: Model 

    #TODO: Self-Supervised Training 

