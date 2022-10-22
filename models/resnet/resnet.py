import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34


__all__ = ['ResNetModel']

class ResNetModel(nn.Module):
    def __init__(self, pretrained=True, head_layers=[512, 512, 512, 512, 512, 512, 512, 512, 128], num_classes=2):
        super(ResNetModel, self).__init__()
        # self.resnet18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=pretrained)
        self.backbone = resnet18(pretrained=pretrained)

        # create MPL head as seen in the code in: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        # TODO: check if this is really the right architecture
        last_layer = 512
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1d(num_neurons))
            sequential_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons

        head = nn.Sequential(
            *sequential_layers
        )
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            head,
            nn.Linear(last_layer, num_classes)
        )

        self.feature_extractor = torch.nn.Sequential(*(list(self.backbone.children())[:7]))
        self.dim_redu = torch.nn.Sequential(*(list(self.backbone.children())[7:9]))

    def forward_features(self, x):
        embeds = self.backbone(x)
        return embeds

    def forward(self, x):
        embeds = self.forward_features(x)
        logits = self.head(embeds)

        dim4_embeds = self.feature_extractor(x)  # (64, 256, 14, 14)
        tmp_embeds = self.dim_redu(dim4_embeds)  # (64, 512, 1, 1)
        dim2_embeds = torch.flatten(tmp_embeds, 1)  #与embeds值相同，(64, 512)

        return embeds, logits

    def freeze_resnet(self):
        # freez full resnet
        for param in self.backbone.parameters():
            param.requires_grad = False
        # unfreeze head:
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        # unfreeze all:
        for param in self.parameters():
            param.requires_grad = True

    def freeze_parameters(self, train_fc=False):
        for p in self.backbone.conv1.parameters():
            p.requires_grad = False
        for p in self.backbone.bn1.parameters():
            p.requires_grad = False
        for p in self.backbone.layer1.parameters():
            p.requires_grad = False
        for p in self.backbone.layer2.parameters():
            p.requires_grad = False
        if not train_fc:
            for p in self.backbone.fc.parameters():
                p.requires_grad = False
