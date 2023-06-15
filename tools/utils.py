import os
import torch
import random
import numpy as np
import torch
import random
import yaml
import time
import shutil
import torchvision

__all__ = ['seed_everything', 'parse_device_list', 'merge_config', 'override_config', 'extract_config', 'create_folders', 
           'record_path', 'save_arg', 'save_log', 'save_script', 'save_image']


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def parse_device_list(device_ids_string, id_choice=None):
    device_ids = [int(i) for i in device_ids_string[0]]
    id_choice = 0 if id_choice is None else id_choice
    device = device_ids[id_choice]
    device = torch.device("cuda", device)
    return device, device_ids

def override_config(previous, new):
    config = previous
    for new_key in new.keys():
        config[new_key] = new[new_key]

    return config

def merge_config(config, args):
    """
    args overlaps config, the args is given a high priority 
    """
    for key_arg in dir(args):
        value = getattr(args, key_arg)
        is_int = (type(value)==int)
        if (getattr(args, key_arg) or is_int) and (key_arg in config.keys()):
            config[key_arg] = getattr(args, key_arg)

    return config

def extract_config(args):
    config = dict()
    for key_arg in vars(args):
        value = getattr(args, key_arg)
        is_int = (type(value)==int)
        if vars(args)[key_arg] or is_int:
            config[key_arg] = vars(args)[key_arg]

    return config

def record_path(para_dict):
    # mkdir ./work_dir/fed/brats/time-dir
    localtime = time.asctime(time.localtime(time.time()))
    file_path = '{}/{}/{}/{}'.format(
        para_dict['work_dir'], para_dict['learning_mode'], para_dict['dataset'], localtime)

    os.makedirs(file_path)

    return file_path

def save_arg(para_dict, file_path):
    with open('{}/config.yaml'.format(file_path), 'w') as f:
        yaml.dump(para_dict, f)

def save_log(infor, file_path, description=None):
    localtime = time.asctime(time.localtime(time.time()))
    infor = '[{}] {}'.format(localtime, infor)

    with open('{}/log{}.txt'.format(file_path, description), 'a') as f:
        print(infor, file=f)

def save_script(src_file, file_path):
    shutil.copy2(src_file, file_path)

def save_image(image, name, image_path):
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    torchvision.utils.save_image(image, '{}/{}'.format(image_path, name), normalize=False)

def create_folders(tag_path):
    if not os.path.exists(tag_path):
        os.makedirs(tag_path)



