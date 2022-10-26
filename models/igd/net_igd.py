from torch import nn
from models.igd.mvtec_module import twoin1Generator256, VisualDiscriminator256

class NetIGD(nn.Module):
    def __init__(self, args):
        super(NetIGD, self).__init__()
        self.args = args

        self.g = twoin1Generator256(64, latent_dimension=self.args._latent_dimension)
        self.d = VisualDiscriminator256(64)