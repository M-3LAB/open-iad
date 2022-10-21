import torch
import torch.nn as nn
from torchvision import models


class feature_resnet18(nn.Module):
    def __init__(self):
        super(feature_resnet18, self).__init__()
        self.net = models.resnet18(pretrained=True)
    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        return x

class feature_resnet50(nn.Module):
    def __init__(self):
        super(feature_resnet50, self).__init__()
        self.net = models.resnet50(pretrained=True)
    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        return x

