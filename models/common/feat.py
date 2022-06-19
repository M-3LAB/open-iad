import torch
import torch.nn as nn

__all__ = ['FeatureExtractor']

class FeatureExtractor(nn.Module):
    def __init__(self, backbone, layers):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        # collect the output dimension from each layer
        self.output_dims = []
        self._features = {layer: torch.empty(0) for layer in self.layers}

        for layer_id in self.layers:
            layer = dict(*self.backbone.named_modules)[layer_id]
            layer.register_forward_hook(self.get_layer_features(layer_id))
            # Get the output dimension 
            layer_modules = [*layer.modules()]
            for idx in reversed(range(len(layer_modules))):
                if hasattr(layer_modules[idx], "out_channels"):
                    self.output_dims.append(layer_modules[idx].out_channels)
                    break 
            

    def get_layer_features(self, layer_id):
        """
        Get Layer Features

        Args:
            layer_id(str)

        Returns:
            layer features
        """
        
        def hook(module, input, output):
            """
            Forward Hook to Extract Features

            Args: 
                output: Feature map collected after the forward-pass
            """
            self._features[layer_id] = output
        
        return hook

    def forward(self, input):
        """_summary_

        Args:
            input (BCHW): input tensor 
            output(torch.tensor): feature map
        """
        self._features = {layer: torch.empty(0) for layer in self.layers}
        _ = self.backbone(input)
        #TODO: Why?
        return self._features

        