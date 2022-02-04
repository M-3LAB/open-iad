import argparse

__all__ = ['parse_argument']

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--dataset', type=str, choices=['mvtec2d', 'mvtec3d', 'mtd'])
    parser.add_argument('--all-classes', action='store_true', default=False)
    parser.add_argument('--class-name', type=str)
    parser.add_argument('--mode', type=str, choices=['centralized', 'continual', 'federated'])
    parser.add_argument('--phase', type=str, choices=['train', 'test', 'ground_truth'])
    parser.add_argument('--backbone-name', type=str, choices=['resnet18', 'wide_resnet50'])
    parser.add_argument('--model', type=str, choices=['patchcore'])
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--num-workers', type=int)
    args = parser.parse_args()
    return args 