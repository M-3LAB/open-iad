import torch
from torch import nn
import torchvision
import torchvision.models as models

DIM = 256
OUTPUT_DIM = 256 * 256 * 3

class MyConvo2d(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True, stride=1, bias=True):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1) / 2)
        self.conv = torch.nn.Conv2d(input_dim, output_dim, kernel_size, stride=stride, padding=self.padding, bias=bias)

    def forward(self, x):
        output = self.conv(x)
        return output


class ConvMeanPool(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init=self.he_init)

    def forward(self, x):
        output = self.conv(x)
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output[:, :, ::2, 1::2] +
                  output[:, :, 1::2, 1::2]) / 4
        return output


class MeanPoolConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init=self.he_init)

    def forward(self, x):
        output = x
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output[:, :, ::2, 1::2]
                  + output[:, :, 1::2, 1::2]) / 4
        output = self.conv(output)
        return output


class DepthToSpace(torch.nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, x):
        output = x.permute(0, 2, 3, 1)
        output = output.contiguous()
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size, input_height, output_width, output_depth) for t_t in spl]
        output = torch.stack(stacks, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, output_height,
                                                                                      output_width, output_depth)
        output = output.contiguous()
        output = output.permute(0, 3, 1, 2)
        output = output.contiguous()
        return output


class UpSampleConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True, bias=True):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init=self.he_init, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, x):
        output = x
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output


class ResidualBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=DIM, encoder=False):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        if resample == 'down':
            if encoder is True:
                self.bn1 = torch.nn.BatchNorm2d(input_dim)
                self.bn2 = torch.nn.BatchNorm2d(input_dim)
            else:
                self.bn1 = torch.nn.LayerNorm([input_dim, hw, hw])
                self.bn2 = torch.nn.LayerNorm([input_dim, hw, hw])
        elif resample == 'up':
            self.bn1 = torch.nn.BatchNorm2d(input_dim)
            self.bn2 = torch.nn.BatchNorm2d(output_dim)
        elif resample is None:
            self.bn1 = torch.nn.BatchNorm2d(output_dim)
            self.bn2 = torch.nn.LayerNorm([input_dim, hw, hw])
        else:
            raise Exception('invalid resample value')

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size=1, he_init=False)
            self.conv_1        = MyConvo2d(input_dim, input_dim, kernel_size=kernel_size, bias=False)
            self.conv_2        = ConvMeanPool(input_dim, output_dim, kernel_size=kernel_size)
        elif resample == 'up':
            self.conv_shortcut = UpSampleConv(input_dim, output_dim, kernel_size=1, he_init=False)
            self.conv_1        = UpSampleConv(input_dim, output_dim, kernel_size=kernel_size, bias=False)
            self.conv_2        = MyConvo2d(output_dim, output_dim, kernel_size=kernel_size)
        elif resample is None:
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size=1, he_init=False)
            self.conv_1        = MyConvo2d(input_dim, input_dim, kernel_size=kernel_size, bias=False)
            self.conv_2        = MyConvo2d(input_dim, output_dim, kernel_size=kernel_size)
        else:
            raise Exception('invalid resample value')

    def forward(self, x):
        if self.input_dim == self.output_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.conv_shortcut(x)

        output = x
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output


# ok
class VisualGenerator(torch.nn.Module):
    def __init__(self, dim=DIM, latent_dimension=250):
        super(VisualGenerator, self).__init__()

        self.dim = dim
        self.ln1 = torch.nn.Linear(latent_dimension, 4 * 4 * (8 * self.dim))
        self.rb1 = ResidualBlock(8 * self.dim, 4 * self.dim, 3, resample='up')
        self.rb2 = ResidualBlock(4 * self.dim, 2 * self.dim, 3, resample='up')
        self.rb3 = ResidualBlock(2 * self.dim, 1 * self.dim, 3, resample='up')
        self.bn = torch.nn.BatchNorm2d(self.dim)

        self.conv1 = MyConvo2d(1 * self.dim, 3, 3)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        output = self.ln1(x.contiguous())
        output = output.view(-1, 8 * self.dim, 4, 4)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)

        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.tanh(output)
        return output


# ok
class VisualDiscriminator(nn.Module):
    def __init__(self, dim=DIM):
        super(VisualDiscriminator, self).__init__()
        self.dim = dim
        self.conv1 = MyConvo2d(3, self.dim, 3, he_init=False)    # 3
        self.rb1 = ResidualBlock(1 * self.dim, 2 * self.dim, 3, resample='down', hw=DIM)
        self.rb2 = ResidualBlock(2 * self.dim, 4 * self.dim, 3, resample='down', hw=int(DIM / 2))
        self.rb3 = ResidualBlock(4 * self.dim, 8 * self.dim, 3, resample='down', hw=int(DIM / 4))
        self.ln1 = nn.Linear(4 * 4 * 8 * self.dim, 1)

    def forward(self, x):
        output = x.contiguous()
        output = output.view(-1, 3, DIM, DIM)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = output.view(-1, 4 * 4 * 8 * self.dim)
        output = self.ln1(output)
        output = output.view(-1)
        return output


class Encoder_256(torch.nn.Module):
    def __init__(self, z_dim=128):
        super(Encoder_256, self).__init__()
        self.dim = 64
        self.conv1 = MyConvo2d(3, self.dim, kernel_size=5, stride=2, he_init=False)    # 3
        self.rb1 = ResidualBlock(1 * self.dim, 2 * self.dim, 3, resample='down', hw=int(DIM / 1), encoder=True)
        self.rb2 = ResidualBlock(2 * self.dim, 4 * self.dim, 3, resample='down', hw=int(DIM / 2), encoder=True)
        self.rb3 = ResidualBlock(4 * self.dim, 8 * self.dim, 3, resample='down', hw=int(DIM / 4), encoder=True)
        self.rb4 = ResidualBlock(8 * self.dim, 8 * self.dim, 3, resample='down', hw=int(DIM / 8), encoder=True)
        self.conv2 = MyConvo2d(8 * self.dim, 8 * self.dim, kernel_size=3, stride=2, he_init=False)
        self.ln1 = nn.Linear(4 * 4 * 8 * self.dim, z_dim)

#
    def forward(self, x):
        output = x.contiguous()
        output = output.view(-1, 3, 256, 256)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = self.conv2(output)
        output = output.view(-1, 4 * 4 * 8 * self.dim)
        output = self.ln1(output)
        return output


# ok
class twoin1Generator256(torch.nn.Module):
    def __init__(self, dim=DIM, latent_dimension=250):
        super(twoin1Generator256, self).__init__()

        self.pretrain = Encoder_256(z_dim=512)

        # path = 'Encoder_256_ckpt'
        # self.pretrain.load_state_dict(torch.load(path))
        # for param in self.pretrain.parameters():
        #     param.requires_grad = False
        # self.z_dim = latent_dimension

        self.pretrain = torchvision.models.resnet18(pretrained=True)
        self.pretrain = nn.Sequential(*list(self.pretrain.children())[:-1])
        for param in self.pretrain.parameters():
            param.requires_grad = False

        # self.pretrain.fc = nn.Linear(512, latent_dimension)
        self.ln11 = nn.Linear(512, latent_dimension)

        self.R = 0.0
        self.c = None
        self.sigma = None
        ############################################################################################################################################

        self.dim = dim
        self.ln1 = torch.nn.Linear(latent_dimension, 4 * 4 * (8 * self.dim))
        self.up1 = UpSampleConv(8 * self.dim, 8 * self.dim, kernel_size=3)
        # self.rb0 = ResidualBlock(8 * self.dim, 8 * self.dim, 3, resample='up')
        self.rb1 = ResidualBlock(8 * self.dim, 8 * self.dim, 3, resample='up')
        self.rb2 = ResidualBlock(8 * self.dim, 4 * self.dim, 3, resample='up')
        self.rb3 = ResidualBlock(4 * self.dim, 2 * self.dim, 3, resample='up')
        self.rb4 = ResidualBlock(2 * self.dim, 1 * self.dim, 3, resample='up')

        self.bn = torch.nn.BatchNorm2d(self.dim)
        # self.conv1 = MyConvo2d(1 * self.dim, 3, 3)
        self.relu = torch.nn.ReLU()
        self.up2 = UpSampleConv(self.dim, 3, kernel_size=3)

        self.tanh = torch.nn.Tanh()
        # self.attn1 = Self_Attn(128, 'relu')

    def encoder(self, x):
        output = self.pretrain(x).view(-1, 512)
        output = self.ln11(output)
        return output

    def generate(self, x):
        output = self.ln1(x)
        output = output.view(-1, 8 * self.dim, 4, 4)

        output = self.up1(output)
        # output = self.rb0(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)

        output = self.bn(output)
        output = self.relu(output)
        output = self.up2(output)
        # output = self.conv1(output)
        return output

    def forward(self, x):
        output = self.encoder(x)
        output = self.generate(output)
        return output


# ok
class VisualDiscriminator256(nn.Module):
    def __init__(self, dim=DIM):
        super(VisualDiscriminator256, self).__init__()
        self.dim = dim
        self.conv1 = MyConvo2d(3, self.dim, kernel_size=5, stride=2, he_init=False)  # 3
        self.rb1 = ResidualBlock(1 * self.dim, 2 * self.dim, 3, resample='down', hw=int(DIM / 2))
        self.rb2 = ResidualBlock(2 * self.dim, 4 * self.dim, 3, resample='down', hw=int(DIM / 4))
        self.rb3 = ResidualBlock(4 * self.dim, 8 * self.dim, 3, resample='down', hw=int(DIM / 8))
        self.rb4 = ResidualBlock(8 * self.dim, 8 * self.dim, 3, resample='down', hw=int(DIM / 16))
        # self.rb5 = ResidualBlock(8 * self.dim, 8 * self.dim, 3, resample='down', hw=int(DIM / 32))
        self.conv2 = MyConvo2d(8 * self.dim, 8 * self.dim, kernel_size=3, stride=2, he_init=False)
        self.ln1 = nn.Linear(4 * 4 * 8 * self.dim, 1)

    def forward(self, x):
        output = x.contiguous()
        output = output.view(-1, 3, DIM, DIM)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = self.conv2(output)
        # output = self.rb5(output)
        output = output.view(-1, 4 * 4 * 8 * self.dim)
        output = self.ln1(output)
        output = output.view(-1)
        return output


################################

class Encoder_128(torch.nn.Module):
    def __init__(self, z_dim=128):
        super(Encoder_128, self).__init__()
        self.dim = 64
        self.conv1 = MyConvo2d(3, self.dim, kernel_size=5, stride=2, he_init=False)    # 3
        self.rb1 = ResidualBlock(1 * self.dim, 2 * self.dim, 3, resample='down', hw=int(DIM / 1), encoder=True)
        self.rb2 = ResidualBlock(2 * self.dim, 4 * self.dim, 3, resample='down', hw=int(DIM / 2), encoder=True)
        self.rb3 = ResidualBlock(4 * self.dim, 8 * self.dim, 3, resample='down', hw=int(DIM / 4), encoder=True)
        self.rb4 = ResidualBlock(8 * self.dim, 8 * self.dim, 3, resample='down', hw=int(DIM / 8), encoder=True)
        self.ln1 = nn.Linear(4 * 4 * 8 * self.dim, z_dim)
#
    def forward(self, x):
        output = x.contiguous()
        output = output.view(-1, 3, 128, 128)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)

        output = output.view(-1, 4 * 4 * 8 * self.dim)
        output = self.ln1(output)
        return output


class Generator_128(torch.nn.Module):
    def __init__(self, dim=64, z_dim=250):
        super(Generator_128, self).__init__()
        self.dim = dim
        self.ln1 = torch.nn.Linear(z_dim, 4 * 4 * (8 * self.dim))
        self.rb1 = ResidualBlock(8 * self.dim, 8 * self.dim, 3, resample='up')
        self.rb2 = ResidualBlock(8 * self.dim, 4 * self.dim, 3, resample='up')
        self.rb3 = ResidualBlock(4 * self.dim, 2 * self.dim, 3, resample='up')
        self.rb4 = ResidualBlock(2 * self.dim, 1 * self.dim, 3, resample='up')
        self.bn = torch.nn.BatchNorm2d(self.dim)
        self.relu = torch.nn.ReLU()
        self.up2 = UpSampleConv(self.dim, 3, kernel_size=5, he_init=False)

    def forward(self, x):
        output = self.ln1(x)
        output = output.view(-1, 8 * self.dim, 4, 4)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = self.bn(output)
        output = self.relu(output)
        output = self.up2(output)
        return output


class twoin1Generator_128(torch.nn.Module):
    def __init__(self, dim=64, latent_dimension=250):
        super(twoin1Generator_128, self).__init__()

        self.pretrain = Encoder_128(z_dim=512)

        path = 'Encoder_128_ckpt'
        self.pretrain.load_state_dict(torch.load(path))
        for param in self.pretrain.parameters():
            param.requires_grad = False
        self.z_dim = latent_dimension

        # self.pretrain = torchvision.models.resnet18(pretrained=True)
        # self.pretrain = nn.Sequential(*list(self.pretrain.children())[:-1])
        # for param in self.pretrain.parameters():
        #     param.requires_grad = False

        # self.pretrain.fc = nn.Linear(512, latent_dimension)
        self.ln11 = nn.Linear(512, latent_dimension)

        self.c = None
        self.sigma = None
        ############################################################################################################################################

        self.decoder = Generator_128(z_dim=latent_dimension)

    def encoder(self, x):
        output = self.pretrain(x).view(-1, 512)
        output = self.ln11(output)
        return output

    def forward(self, x):
        output = self.encoder(x)
        output = self.decoder(output)
        return output