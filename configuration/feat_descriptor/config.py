import argparse

__all__ = ['parse_arguments_feat_descriptor']

def parser_arguments_feat_descriptor():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default=None, choices=['mvtec3d', 'model40'])
    parser.add_argument('--feat', '-f', type=str, default=None, choices=['neuralpoint_dgcnn', 'neuralpoint_pointmlp'])
    parser.add_argument('--num-affinity-points', type=int, default=None, help='the required number of affinity points for feature calculation ')
    parser.add_argument('--distance', type=str, choices=['cd','dcd','emd','normal'])
    parser.add_argument('--method', type=str)
    args = parser.parse_args()
    return args