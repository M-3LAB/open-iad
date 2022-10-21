import argparse
import numpy as np
import torch
import random
import yaml
import re

class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):
        raise AttributeError(
            f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")

def set_deterministic(seed):
    # seed by default is None
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', default='./configs/dis.yaml', type=str, help="xxx.yaml")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_dir', type=str, default="/home/wwjxk10/code_python/datasets/defect_datasets/mvtec")
    parser.add_argument('--mtd_dir', type=str, default="/home/wwjxk10/code_python/datasets/defect_datasets/mtd_ano_mask")
    parser.add_argument('--anomaly_source_path', type=str, default='/home/wwjxk10/code_python/datasets/dtd/images')
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        print(data)
        for key, value in Namespace(data).__dict__.items():
            vars(args)[key] = value

    set_deterministic(args.seed)

    return args