import argparse

__all__ = ['parse_arguments_feat_descriptor']

def parser_arguments_feat_descriptor():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default=None, choices=['mvtec3d', 'model40'])
    parser.add_argument('--feat', '-f', type=str, default=None, choices=['neuralpoint_dgcnn', 'neuralpoint_pointmlp'])
    args = parser.parse_args()
    return args