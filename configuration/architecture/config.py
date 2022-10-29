import argparse
from asyncio import FastChildWatcher
from logging import root
import socket,fcntl,struct

def get_ip_address(ifname):
    s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    info = fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', bytes(ifname[:15], 'utf-8')))

    return socket.inet_ntoa(info[20:24])

__all__ = ['parse_arguments_centralized', 'parse_arguments_federated']

def assign_service(guoyang):
    
    if guoyang:
        ip = get_ip_address('lo')
    else:
        ip = get_ip_address('eno1')

    root_path = None

    if ip == '172.18.36.46':
        root_path = '/disk4/xgy' 
    if ip == '127.0.0.1':
        root_path = '/home/robot/data'
    if ip == '172.18.34.25':
        root_path = '/home/zhengf_lab/cse30010351/m3lab/data'
    if ip == '172.18.36.107':
        root_path = '/ssd-sata1/wjb/data/open-ad'
    if ip == '172.18.36.108':
        root_path = '/ssd2/m3lab/data/open-ad'

    return ip, root_path

def parse_arguments_centralized():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='mvtec2d', choices=['mvtec2d', 'mvtec3d', 'mpdd', 'mvtecloco', 'mtd', 'btad', 'mvtec2df3d'])
    parser.add_argument('--model', '-m', type=str, default='spade', choices=['patchcore', 'csflow', 'dne', 'draem', 'igd', 'cutpaste', 'devnet', 'dra', 
                                                                              'favae', 'padim', 'reverse', 'spade', 'fastflow', 'softpatch', 'cfa'])
    parser.add_argument('--net', '-n', type=str, default='resnet18', choices=['wide_resnet50', 'resnet18', 'net_csflow', 'vit_b_16', 'net_draem', 'net_dra',
                                                                              'net_igd', 'net_reverse'])
    parser.add_argument('--root-path', '-rp', type=str, default=None)
    parser.add_argument('--data-path', '-dp', type=str, default=None)

    parser.add_argument('--train-task-id', '-tid', type=int, default=[0], nargs='+')
    parser.add_argument('--valid-task-id', '-vid', type=int, default=[0], nargs='+')
    parser.add_argument('--coreset-sampling-ratio', '-csr', type=float, default= 0.0001)

    # vanilla learning
    parser.add_argument('--vanilla', '-v', action='store_true', default=True)
    
    # semi-supervised learning
    parser.add_argument('--semi', '-s', action='store_true', default=False)
    parser.add_argument('--semi-anomaly-num', '-san', type=int, default=5)
    parser.add_argument('--semi-overlap', '-so', action='store_true', default=False)
    
    # continual learning
    parser.add_argument('--continual', '-c', action='store_true', default=False)

    # fewshot learniing
    parser.add_argument('--fewshot', '-f', action='store_true', default=False)
    parser.add_argument('--fewshot-exm', '-fe', type=int, default=1)
    parser.add_argument('--fewshot-data-aug', '-fda', action='store_true', default=False)
    parser.add_argument('--fewshot-feat-aug', '-ffa', action='store_true', default=False)
    parser.add_argument('--fewshot-num-dg', '-fnd', type=int, default=1)

    # noisy label
    parser.add_argument('--noisy', '-ny', action='store_true', default=False)
    parser.add_argument('--noisy-overlap', '-no', action='store_true', default=False)
    parser.add_argument('--noisy-ratio', '-nr', type=float, default=0.1)

    parser.add_argument('--gpu-id', '-g', type=str, default=2)
    parser.add_argument('--num-epoch', '-ne', type=int, default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--vis-em', action='store_true', default=False)
    parser.add_argument('--guoyang', '-gy', action='store_true', default=False)

    # data augmentation type
    parser.add_argument('--train-aug-type', '-tag', choices=['normal', 'cutpaste'], help='data augmentation type')
    parser.add_argument('--valid-aug-type', '-vag', choices=['normal', 'cutpaste'], help='data augmentation type')

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
    