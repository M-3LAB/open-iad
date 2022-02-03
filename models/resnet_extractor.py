import torch
import torch.nn as nn
from torchvision import models

__all__ = ['ResNetExtractor']

class ResNetExtractor(nn.Module):
    def __init__(self, device, backbone_name='resnet18'):
        super(ResNetExtractor).__init__()
        assert backbone_name in ['resnet18', 'wide_resnet50'], 'Not Implemented Yet' 

        self.device = device

        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=True, progress=True).to(self.device)
        elif backbone_name == 'wide_resnet50':
            self.backbone = models.wide_resnet50_2(pretrained=True, progress=True).to(self.device)
        else:
            raise NotImplementedError('This Pretrained Model Not Implemented Error')
        
        self.backbone.eval()
    
    def forward(self, x, channel_indices, layer_hook, layer_indices):

        """
        Run Inference on backbone and return the embedding vectors
        
        Args:
            batch: A batch of images
            channel_indices: A list of indices with the desired channels to include in
                the embedding vectors.
            layer_hook: A function that runs on each layer of the resnet before
                concatenating them.
            layer_indices: A list of indices with the desired layers to include in the
                embedding vectors.

        Returns:
            embedding_vectors: The embedding vectors.
        """
        with torch.no_grad():
            
            # stem
            x = self.backbone.conv1(x) 
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            layer1 = self.backbone.layer1(x)
            layer2 = self.backbone.layer2(layer1)
            layer3 = self.backbone.layer3(layer2)
            layer4 = self.backbone.layer4(layer3)
            layers = [layer1, layer2, layer3, layer4]

            if layer_indices is not None:
                layers = [layers[i] for i in layer_indices]

            if layer_hook is not None:
                layers = [layer_hook(layer) for layer in layers]