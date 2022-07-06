import argparse

__all__ = ['parse_arguments_centralized', 'parse_arguments_federated']


def parse_arguments_centralized():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='mvtec2d', choices=['mvtec2d', 'mvtec3d', 'mtd'])
    parser.add_argument('--model', '-m', type=str, default='patchcore2d', choices=['patchcore2d', 'patchcore3d'])
    parser.add_argument('--data-path', '-dp', type=str, default=None)

    parser.add_argument('--chosen-train-task-ids', type=int, default=3, nargs='+')
    parser.add_argument('--chosen-test-task-id', type=int, default=None)

    parser.add_argument('--continual', '-conti', action='store_true', default=False)
    parser.add_argument('--num-task', type=int, default=15)
    parser.add_argument('--fewshot', action='store_true', default=None)
    parser.add_argument('--fewshot-exm', type=int, default=5)

    parser.add_argument('--gpu-id', '-g', type=str, default=None)
    parser.add_argument('--num-epoch', type=int, default=None)
    parser.add_argument('--debug', action='store_true', default=False)

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
    