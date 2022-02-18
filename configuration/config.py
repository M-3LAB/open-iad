import argparse
from this import d

__all__ = ['parse_argument']

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='/disk1/mvtec/2D')
    parser.add_argument('--dataset', type=str, default='mvtec2d', choices=['mvtec2d', 'mvtec3d', 'mtd'])
    parser.add_argument('--all-classes', action='store_true', default=False)
    parser.add_argument('--class-name', type=str)
    parser.add_argument('--mode', type=str, default='continual', choices=['centralized', 'continual', 'federated'])
    parser.add_argument('--num-tasks-continual', type=int, default=5)
    parser.add_argument('--phase', type=str, choices=['train', 'test', 'ground_truth'])
    parser.add_argument('--backbone-name', type=str, choices=['resnet18', 'wide_resnet50'])
    parser.add_argument('--model', type=str, choices=['patchcore'])
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--layer-hook', default=None, help='')
    parser.add_argument('--layer-indicies', default=[1,2], help='')
    parser.add_argument('--lr', type=float, help='learning rate')
    args = parser.parse_args()
    return args 