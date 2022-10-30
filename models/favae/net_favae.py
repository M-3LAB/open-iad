import torch
from torch import nn
#from torchsummary import summary

__all__ = ['NetFAVAE']

class NetFAVAE(nn.Module):
    def __init__(self, input_channel=3, z_dim=100):
        super(NetFAVAE, self).__init__()

        # encode
        self.encode = nn.Sequential(
            nn.Conv2d(input_channel, 128, kernel_size=4, stride=2, padding=1),  # 128 => 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),  # 64 => 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),  # 32 => 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 16 => 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(512, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 200, kernel_size=8, stride=1),  # 8 => 1
            nn.Flatten(),
            Split())

        # decode
        self.decode = nn.Sequential(
            DeFlatten(),
            nn.ConvTranspose2d(100, 32, kernel_size=8, stride=1),  # 1 => 8
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(32, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # 8 => 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # 16 => 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),  # 32 => 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(128, input_channel, kernel_size=4, stride=2, padding=1),  # 64 => 128
            nn.Identity(),
            nn.Sigmoid()
            # nn.Tanh()
        )

        self.adapter = nn.ModuleList([Adapter_model(128), Adapter_model(256), Adapter_model(512)])

    def reparameterize(self, mu, log_var):
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return z, self.decode(z), mu, logvar


class DeFlatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], 100, 1, 1)


class Split(nn.Module):
    def forward(self, x):
        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar


class Adapter_model(nn.Module):
    def __init__(self, channel=128):
        super(Adapter_model, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=1, stride=1), nn.ReLU(),
                                  nn.Conv2d(channel, channel, kernel_size=1, stride=1))

    def forward(self, x):
        return self.conv(x)