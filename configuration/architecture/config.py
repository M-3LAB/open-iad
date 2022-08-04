import argparse
from logging import root
import socket

__all__ = ['parse_arguments_centralized', 'parse_arguments_federated']

def assign_service():
    host_name = socket.gethostname()
    ip = socket.gethostbyname(host_name)
    
    root_path = None
    if ip == '172.18.36.46':
        root_path = '/disk4/xgy' 
    elif ip == '192.168.1.104':
        root_path = '/home/robot/data'
    else:
        raise 'Load Dataset Error'

    return ip, root_path

def parse_arguments_centralized():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='mvtec2d', choices=['mvtec2d', 'mvtec3d', 'mpdd', 'mvteclogical'])
    parser.add_argument('--model', '-m', type=str, default='patchcore2d', choices=['patchcore2d', 'reverse',
                                                                                   'spade', 'padim2d', 'stpm',
                                                                                   'cfa'])
    parser.add_argument('--root-path', '-rp', type=str, default=None)
    parser.add_argument('--data-path', '-dp', type=str, default=None)

    parser.add_argument('--chosen-train-task-ids', type=int, default=[2], nargs='+')
    parser.add_argument('--chosen-test-task-id', type=int, default=2)
    parser.add_argument('--coreset-sampling-ratio', type=float, default=1)

    parser.add_argument('--fewshot', action='store_true', default=False)
    parser.add_argument('--fewshot-normal', action='store_true', default=True)
    parser.add_argument('--num-dg', type=int, default=1)
    parser.add_argument('--fewshot-exm', type=int, default=1)

    parser.add_argument('--continual', '-conti', action='store_true', default=False)
    parser.add_argument('--gpu-id', '-g', type=str, default=1)
    parser.add_argument('--num-epoch', type=int, default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--vis-em', action='store_true', default=False)

    parser.add_argument('--data-aug', '-da', action='store_true', default=False)
    parser.add_argument('--feat-aug', '-fa', action='store_true', default=False)

    #parser.add_argument('--save-model', action='store_true', default=False)
    #parser.add_argument('--load-model', action='store_true', default=False)
    #parser.add_argument('--load-model-dir', type=str, default=None)


    args = parser.parse_args()
    return args

def parse_arguments_federated():
    parser = argparse.ArgumentParser()
    # federated setting
    parser.add_argument('--fed-aggregate-method', '-fam', type=str, default=None)
    parser.add_argument('--num-round', type=int, default=None)

    # centralized setting
    # TODO

    args = parser.parse_args()
    return args
    