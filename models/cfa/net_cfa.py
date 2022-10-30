from torch import nn
from models.cfa.resnet import resnet18

class NetCFA(nn.Module):
    def __init__(self, args):
        super(NetCFA, self).__init__()
        self.args = args

        self.resnet18 = resnet18(pretrained=True, progress=True)