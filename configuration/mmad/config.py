import argparse

__all__ = ['parse_arguments_mmad']

def parse_arguments_mmad():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--class-name', default=None, type=str)
    parser.add_argument('--batch-size', default=None, type=int)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--gpu-id', default=None, type=int)
    parser.add_argument('--num-epochs', default=None, type=int)
    parser.add_argument('--data-path', default=None, type=str)
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--fusion-method', default=None, type=str)
    parser.add_argument('--cl', action='store_true', default=False)
    
    args = parser.parse_args()
    return args