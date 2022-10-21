import torch
import torch.nn as nn
from configuration.feat_descriptor.config import parser_arguments_feat_descriptor
from tools.utils import *
import yaml
from models.pointcore.descriptor.neural_point.neural_point import NeuralPoint


if __name__ == '__main__':
    args = parser_arguments_feat_descriptor()

    with open('./configuration/feat_descriptor/{}.yaml'.format(args.dataset), 'r') as f:
        para_dict = yaml.load(f, Loader=yaml.SafeLoader)

    para_dict = merge_config(para_dict, args)
    print(para_dict)

    #TODO: Data Loader

    #TODO: Model 

    #TODO: Self-Supervised Training 