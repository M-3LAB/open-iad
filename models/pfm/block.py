import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101
from torchvision.models.vgg import vgg16_bn, vgg19_bn

class PretrainedModel(nn.Module):
    def __init__(self, model_name):
        super(PretrainedModel, self).__init__()
        if "resnet" in model_name:
            model = eval(model_name)(pretrained=True)
            modules = list(model.children())
            self.block1 = nn.Sequential(*modules[0:4])
            self.block2 = modules[4]
            self.block3 = modules[5]
            self.block4 = modules[6]
            self.block5 = modules[7]
        elif "vgg" in model_name:
            if model_name == "vgg16_bn":
                self.block1 = nn.Sequential(*self.modules[0:14])
                self.block2 = nn.Sequential(*self.modules[14:23])
                self.block3 = nn.Sequential(*self.modules[23:33])
                self.block4 = nn.Sequential(*self.modules[33:43])
            else:
                self.block1 = nn.Sequential(*self.modules[0:14])
                self.block2 = nn.Sequential(*self.modules[14:26])
                self.block3 = nn.Sequential(*self.modules[26:39])
                self.block4 = nn.Sequential(*self.modules[39:52])
        else:
            raise NotImplementedError

    def forward(self, x):
        # B x 64 x 64 x 64
        out1 = self.block1(x)
        # B x 128 x 32 x 32
        out2 = self.block2(out1)
        # B x 256 x 16 x 16
        # 32x32x128
        out3 = self.block3(out2)
        # 16x16x256
        out4 = self.block4(out3)
        return {"out2": out2,
                "out3": out3,
                "out4": out4
                }

class Conv_BN_Relu(nn.Module):
    def __init__(self, in_dim, out_dim, k=1, s=1, p=0, bn=True, relu=True):
        super(Conv_BN_Relu, self).__init__()
        self.conv = [
            nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p),
        ]
        if bn:
            self.conv.append(nn.BatchNorm2d(out_dim))
        if relu:
            self.conv.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)