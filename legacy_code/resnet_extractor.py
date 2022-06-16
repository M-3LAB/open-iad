import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F 

__all__ = ['ResNetExtractor']

def concatenate_two_layers(layer1, layer2):
    """
    Scale the second tensor to the height and width of the first tensor and concatenate them
    Args:
        layer1: torch.Tensor
        layer2: torch.Tensor
    
    Return torch.Tensor
    """

    device = layer1.device
    batch_length, channel_num1, height1, width1 = layer1.size()
    _, channel_num2, height2, width2 = layer2.size()
    height_ratio = int(height1 / height2)
    layer1 = F.unfold(layer1, kernel_size=height_ratio, dilation=1, stride=height_ratio)
    layer1 = layer1.view(batch_length, channel_num1, -1, height2, width2)
    result = torch.zeros(batch_length, channel_num1 + channel_num2, layer1.size(2),
                         height2, width2, device=device)
    for i in range(layer1.size(2)):
        result[:, :, i, :, :] = torch.cat((layer1[:, :, i, :, :], layer2), 1)
    del layer1
    del layer2
    result = result.view(batch_length, -1, height2 * width2)
    result = F.fold(result, kernel_size=height_ratio,
                    output_size=(height1, width1), stride=height_ratio)
    return result

def concatenate_layers(layers):
    """Scale all tensors to the heigth and width of the first tensor and concatenate them."""

    expanded_layers = layers[0]
    for layer in layers[1:]:
        expanded_layers = concatenate_two_layers(expanded_layers, layer)
    return expanded_layers

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
    
    def forward(self, x, channel_indices=None, layer_hook=None, layer_indices=None):

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

            embedding_vectors = concatenate_layers(layers) 

            if channel_indices is not None:
                embedding_vectors = torch.index_select(embedding_vectors, 1, channel_indices)
            
            batch_size, length, width, height = embedding_vectors.shape
            embedding_vectors = embedding_vectors.reshape(batch_size, length, width*height)
            embedding_vectors = embedding_vectors.permute(0, 2, 1)

            return embedding_vectors