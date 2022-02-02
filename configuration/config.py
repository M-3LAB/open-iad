import argparse

__all__ = ['parse_argument']

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['centralized', 'continual', 'federated'])
    parser.add_argument('--phase', type=str, choices=['train', 'test'])
    args = parser.parse_args()
    return args 