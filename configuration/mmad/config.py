import argparse

__all__ = ['parse_arguments_mmad']

def parse_arguments_mmad():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--class-names', default=None, type=str)
    parser.add_argument('--batch-size', default=None, type=int)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--gpu-id', default=None, type=int)
    parser.add_argument('--num-epochs', default=None, type=int)
    parser.add_argument('--data-path', default=None, type=str)
    parser.add_argument('--dataset', default='mvtec3d', type=str, choices=['mvtec3d'])
    parser.add_argument('--fusion-method', default=None, type=str)
    parser.add_argument('--cl', action='store_true', default=None)
    parser.add_argument('--depth-duplicate', action='store_true', default=None)
    
    args = parser.parse_args()
    return args