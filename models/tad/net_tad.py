import torch
import torch.nn as nn

__all__ = ['NetTAD']

class NetTAD(nn.Module):
    def __init__(self, args):
        super(NetTAD, self).__init__()
        self.args = args
