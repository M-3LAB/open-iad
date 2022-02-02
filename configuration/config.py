import argparse

__all__ = ['parse_argument']

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument()
    args = parser.parse_args()
    return args 