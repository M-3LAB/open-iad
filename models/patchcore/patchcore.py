import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.common.feat import FeatureExtractor
from models.patchcore.anomaly_map import AnomalyMapGenerator

__all__ = ['PatchCore']

class PatchCore(nn.Module):

    def __init__(self, input_size, backbone_name, device, layers, num_neighbours):

        super(PatchCore).__init__()

        """
        Args:
            backbone: The name of the desired backbone, i.e., ['resnet18', 'wide_resnet'] 
            device: GPU
        """

        self.device = device
        self.backbone_name = backbone_name
        self.layers = layers
        self.input_size = input_size
        self.num_neighbours = num_neighbours
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=self.input_size)

        self.feature_extractor = FeatureExtractor(backbone=self.bachbone(pretrained=True), layers=self.layers) 
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)

        # Register_buffer collect the one that does not belong to model into state_dict
        self.register_buffer("memory_bank", torch.Tensor())
        self.memory_bank: torch.Tensor
    
    def generate_embedding(self, features):
        """Generate embedding from hierarchical feature map

        Args:
            features (Dict): torch tensor 
        """
        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            #TODO: Why Need to Interpolate
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), dim=1)

        return embeddings
    
    @staticmethod
    def reshape_embedding(embedding_tensor):
        """
        Reshape Embedding from [batch, embedding, patch, patch] to 
        [batch*patch*patch, embedding]
        """
        embedding_size = embedding_tensor.size(1)
        embedding_tensor = embedding_tensor.permute(0, 2, 3, 1).reshape(-1, embedding_size)
        return embedding_tensor
    
    def subsample_embedding(self, embedding, sample_ratio):
        pass

    def nearest_neighbors(self, embedding, n_neighbors):
        """ Nearest Neighbour Search

        Args:
            embedding (Tensor): Features to comprare the distance with the memory bank 
            n_neighbors (int): Number of Neighbours 
        
        Returns:
            Patch scores: (Tensor)
        """
        # Euclidean norm between embedding and memory bank
        #TODO: Why
        distances = torch.cdist(embedding, self.memory_bank, p=2.0)
        patch_scores, _ = distances.topk(k=n_neighbors, largest=False, dim=1)

        return patch_scores
    
    def subsample_embedding(self):
        pass
        
        
    def forward(self, x):

        """
        Return embedding during training
        """
        with torch.no_grad():
            features = self.feature_extractor(x)

        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)

        # BCHW format
        feature_map_shape = embedding.shape[-2:] 
        embedding = self.reshape_embedding(embedding_tensor=embedding)

        if self.training:
            output = embedding
        else:
            patch_scores = self.nearest_neighbors(embedding, n_neighbors=self.num_neighbours)
            anomaly_map, anomaly_score = self.anomaly_map_generator(
                patch_scores=patch_scores, feature_map_shape=feature_map_shape
            )
            output = (anomaly_map, anomaly_score) 
        
        return output

        
