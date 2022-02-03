import argparse

__all__ = ['parse_argument']

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['centralized', 'continual', 'federated'])
    parser.add_argument('--phase', type=str, choices=['train', 'test', 'ground_truth'])
    parser.add_argument('--backbone-name', type=str, choices=['resnet18', 'wide_resnet50'])
    args = parser.parse_args()
    return args 