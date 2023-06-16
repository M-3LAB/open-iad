from torch import nn

class NetExample(nn.Module):
    def __init__(self, args):
        super(NetExample, self).__init__()
        self.args = args