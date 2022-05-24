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
    parser.add_argument('--fusion-method', default=None, type=str, choices=['late', 'early', 'middle'])
    parser.add_argument('--cl', action='store_true', default=None, help='continuous learning mode or not')
    parser.add_argument('--depth-duplicate', type=int, default=1, choices=[1, 3])
    parser.add_argument('--ck-path', default=None, help='checkpoint path')
    
    args = parser.parse_args()
    return args