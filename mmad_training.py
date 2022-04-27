import torch
import torch.nn as nn
import yaml
from configuration.mmad.config import parse_arguments_mmad 
from tools.utilize import *

if __name__ == '__main__':

    args = parse_arguments_mmad() 

    with open('./configuration/mmad/{}.yaml'.format(args.dataset), 'r') as f:
        para_dict = yaml.load(f, Loader=yaml.SafeLoader)
    
    para_dict = merge_config(para_dict, args)
    print(para_dict)

    #TODO: Data Loader

