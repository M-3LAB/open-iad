from torchvision.models import alexnet
from modeling.networks.resnet18 import FeatureRESNET18

NET_OUT_DIM = {'alexnet': 256, 'resnet18': 512}

def build_feature_extractor(backbone, cfg):
    if backbone == "alexnet":
        print("Feature extractor: AlexNet")
        return alexnet(pretrained=True).features
    elif backbone == "resnet18":
        print("Feature extractor: ResNet-18")
        return FeatureRESNET18()
    else:
        raise NotImplementedError