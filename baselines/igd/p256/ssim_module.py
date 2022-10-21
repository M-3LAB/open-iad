import torch
from torch import nn
import torch.nn.functional as F
DIM = 32
OUTPUT_DIM = 32 * 32 * 3

class MyConvo2d(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True, stride=1, bias=True):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1) / 2)
        self.conv = torch.nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=self.padding, bias=bias)

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


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        feature_map = out
        output = self.gamma * out + x
        return output, attention, feature_map


class Encoder(torch.nn.Module):
    def __init__(self, z_dim=512):
        super(Encoder, self).__init__()
        self.dim = 64
        self.conv1 = MyConvo2d(3, self.dim, 3, he_init=False)    # 3
        self.rb1 = ResidualBlock(    self.dim, 2 * self.dim, 3, resample='down', hw=DIM         , encoder=True)
        self.rb2 = ResidualBlock(2 * self.dim, 4 * self.dim, 3, resample='down', hw=int(DIM / 2), encoder=True)
        self.rb3 = ResidualBlock(4 * self.dim, 8 * self.dim, 3, resample='down', hw=int(DIM / 4), encoder=True)
        self.ln1 = nn.Linear(4 * 4 * 8 * self.dim, z_dim)

    def forward(self, x):
        output = x.contiguous()
        output = output.view(-1, 3, 32, 32)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = output.view(-1, 4 * 4 * 8 * self.dim)
        output = self.ln1(output)
        # output = self.tanh(output)
        return output


class DSVDDEncoder(torch.nn.Module):
    def __init__(self, z_dim=128):
        super(DSVDDEncoder, self).__init__()
        self.pretrain = Encoder(z_dim=512)
        self.pretrain.load_state_dict(torch.load('./check_points/teacher_En'))
        # for param in self.pretrain.parameters():
        #     param.requires_grad = False
        self.ln11 = nn.Linear(512, 128)

        self.R = 0.0
        self.c = None

    def forward(self, x):
        z = self.pretrain(x).view(-1, 512)
        output = F.relu(self.ln11(z))
        return output


# ok
class DSVDDGenerator(torch.nn.Module):
    def __init__(self, dim=DIM, latent_dimension=250):
        super(DSVDDGenerator, self).__init__()

        self.dim = dim
        self.ln1 = torch.nn.Linear(latent_dimension, 4 * 4 * (8 * self.dim))
        self.rb1 = ResidualBlock(8 * self.dim, 4 * self.dim, 3, resample='up')
        self.rb2 = ResidualBlock(4 * self.dim, 2 * self.dim, 3, resample='up')
        self.rb3 = ResidualBlock(2 * self.dim, 1 * self.dim, 3, resample='up')
        self.bn = torch.nn.BatchNorm2d(self.dim)
        self.conv1 = MyConvo2d(1 * self.dim, 3, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output = self.ln1(x.contiguous())
        output = output.view(-1, 8 * self.dim, 4, 4)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)

        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        return output


import torch.nn.functional as F
class CIFAR10_LeNet_ELU(nn.Module):
    def __init__(self, z_dim = 128):
        super().__init__()

        self.rep_dim = z_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.elu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.elu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.elu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


# ok
class twoin1Generator(torch.nn.Module):
    def __init__(self, dim=DIM, latent_dimension=250):
        super(twoin1Generator, self).__init__()

        self.pretrain = Encoder(z_dim=512)

        # path = '/home/user/Desktop/user/cifar10_exp/EN500000/EN500000_ckpt'
        # path = './check_points/teacher_En'
        # self.pretrain.load_state_dict(torch.load(path))
        # for param in self.pretrain.parameters():
        #     param.requires_grad = False

        # self.encoder = torchvision.models.resnet18(pretrained=True)
        # self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.ln11 = nn.Linear(512, latent_dimension)

        self.R = 0.0
        self.c = None
        # self.sigma = nn.Parameter(100*torch.ones(1))

        ############################################################################################################################################

        self.dim = dim
        self.ln1 = torch.nn.Linear(latent_dimension, 4 * 4 * (8 * self.dim))
        self.rb1 = ResidualBlock(8 * self.dim, 4 * self.dim, 3, resample='up')
        self.rb2 = ResidualBlock(4 * self.dim, 2 * self.dim, 3, resample='up')
        self.rb3 = ResidualBlock(2 * self.dim, 1 * self.dim, 3, resample='up')
        self.bn = torch.nn.BatchNorm2d(self.dim)

        self.conv1 = MyConvo2d(1 * self.dim, 3, 3)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        # self.attn1 = Self_Attn(128, 'relu')


    def encoder(self, x):
        z = self.pretrain(x)#.view(-1, 512)
        output = self.ln11(z)
        return output

    def generate(self, x):
        # output = self.encoder(x)

        output = self.ln1(x)
        output = output.view(-1, 8 * self.dim, 4, 4)
        output = self.rb1(output)
        output = self.rb2(output)
        # output, attention, feature_map = self.attn1(output)
        output = self.rb3(output)

        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        return output

    def forward(self, x):
        output = self.encoder(x)
        output = self.generate(output.clone())
        return output


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
        # self.tanh = torch.nn.Tanh()

    def forward(self, x):
        output = self.ln1(x.contiguous())
        output = output.view(-1, 8 * self.dim, 4, 4)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)

        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        return output


class VisualDiscriminator(nn.Module):
    def __init__(self, dim=DIM):
        super(VisualDiscriminator, self).__init__()

        self.dim = dim
        self.conv1 = MyConvo2d(3, self.dim, 3, he_init=False)    # 3
        self.rb1 = ResidualBlock(1 * self.dim, 2 * self.dim, 3, resample='down', hw=DIM)
        self.rb2 = ResidualBlock(2 * self.dim, 4 * self.dim, 3, resample='down', hw=int(DIM / 2))
        self.rb3 = ResidualBlock(4 * self.dim, 8 * self.dim, 3, resample='down', hw=int(DIM / 4))
        # self.rb4 = ResidualBlock(8 * self.dim, 8 * self.dim, 3, resample='down', hw=int(DIM / 8))
        self.ln1 = nn.Linear(4 * 4 * 8 * self.dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # output = x.contiguous()
        # output = output.view(-1, 3, DIM, DIM)
        output = self.conv1(x)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        # output = self.rb4(output)
        output = output.view(-1, 4 * 4 * 8 * self.dim)
        output = self.ln1(output)
        # output = output.view(-1, 1)
        # output = self.sigmoid(output)
        return output


