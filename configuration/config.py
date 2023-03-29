import argparse
from asyncio import FastChildWatcher
from logging import root

__all__ = ['parse_arguments_main']

def parse_arguments_main():
    parser = argparse.ArgumentParser()
    ## learning paradigm 
    parser.add_argument('--paradigm', '-p', type=str, default='c2d', choices=['c2d', 'c3d', 'f2d'])

    # ----------------------------- centralized learning ----------------------------- #
    parser.add_argument('--dataset', '-d', type=str, default='mvtec2d', choices=['mvtec2d', 'mvtec3d', 'mpdd', 'mvtecloco', 'mtd', 'btad', 'mvtec2df3d', 'visa', 'dagm', 'imad_hardware_parts'])

    parser.add_argument('--model', '-m', type=str, default='patchcore', choices=['_patchcore', 'patchcore', 'csflow', 'dne', 'draem', 'igd', 'cutpaste', 'devnet', 'dra', 
                                                                              'favae', 'padim', 'reverse', 'spade', 'fastflow', 'softpatch', 'cfa', 'stpm', 'graphcore'])
    parser.add_argument('--net', '-n', type=str, default='resnet18', choices=['wide_resnet50', 'resnet18', 'net_csflow', 'vit_b_16', 'net_draem', 'net_dra',
                                                                              'net_igd', 'net_reverse', 'net_favae', 'net_fastflow', 'net_cfa', 'net_devnet',
                                                                              'vig_ti_224_gelu', 'vig_s_224_gelu', 'vig_b_224_gelu']) 
                                                                              #'pvig_ti_224_gelu', 'pvig_s_224_gelu', 'pvig_m_224_gelu', 'pvig_b_224_gelu'])
    parser.add_argument('--root-path', '-rp', type=str, default=None)
    parser.add_argument('--data-path', '-dp', type=str, default=None)

    parser.add_argument('--train-task-id', '-tid', type=int, default=[11], nargs='+')
    parser.add_argument('--valid-task-id', '-vid', type=int, default=[11], nargs='+')
    parser.add_argument('--sampler-percentage', '-sp', type=float, default= 0.01)

    # vanilla learning
    parser.add_argument('--vanilla', '-v', action='store_true', default=False)
    
    # semi-supervised learning
    parser.add_argument('--semi', '-s', action='store_true', default=False)
    parser.add_argument('--semi-anomaly-num', '-san', type=int, default=5)
    parser.add_argument('--semi-overlap', '-so', action='store_true', default=False)
    
    # continual learning
    parser.add_argument('--continual', '-c', action='store_true', default=False)

    # transfer AD
    parser.add_argument('--transfer', '-tr', action='store_true', default=False)

    # testing-time-training AD
    parser.add_argument('--testing-time-training', '-ttt', action='store_true', default=False)

    # logical AD
    parser.add_argument('--num-parallel-workers', '-npw', type=int, default=8)

    # fewshot learning
    parser.add_argument('--fewshot', '-f', action='store_true', default=False)
    parser.add_argument('--fewshot-exm', '-fe', type=int, default=1)
    parser.add_argument('--fewshot-data-aug', '-fda', action='store_true', default=False)
    parser.add_argument('--fewshot-feat-aug', '-ffa', action='store_true', default=False)
    parser.add_argument('--fewshot-num-dg', '-fnd', type=int, default=1)
    parser.add_argument('--fewshot-aug-type', '-fat', choices=['normal', 'rotation', 'scale', 'translate', 'flip', 'color_jitter', 'perspective'], default=['normal'], nargs='+')

    # noisy label
    parser.add_argument('--noisy', '-z', action='store_true', default=False)
    parser.add_argument('--noisy-overlap', '-no', action='store_true', default=False)
    parser.add_argument('--noisy-ratio', '-nr', type=float, default=0.1)

    parser.add_argument('--gpu-id', '-g', type=str, default=1)
    parser.add_argument('--num-epochs', '-ne', type=int, default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--vis', action='store_true', default=True)
    parser.add_argument('--vis-em', action='store_true', default=False)
    parser.add_argument('--guoyang', '-gy', action='store_true', default=False)

    # data augmentation type
    parser.add_argument('--train-aug-type', '-tag', choices=['normal', 'cutpaste'], help='data augmentation type')
    parser.add_argument('--valid-aug-type', '-vag', choices=['normal', 'cutpaste'], help='data augmentation type')

    # not used
    parser.add_argument('--save-model', action='store_true', default=False)
    parser.add_argument('--load-model', action='store_true', default=False)
    parser.add_argument('--load-model-dir', type=str, default=None)

    # ----------------------------- federated learning ----------------------------- #
    parser.add_argument('--fed-aggregate-method', '-fam', type=str, default=None)
    parser.add_argument('--num-round', type=int, default=None)

    # TODO

    args = parser.parse_args()
    return args
    
