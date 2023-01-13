import torch
from feature_extractors.features import Features
from utils.DenseSIFTDescriptor import  DenseSIFTDescriptor

class SIFTFeatures(Features):

    def __init__(self):
        super().__init__()
        self.dsift = DenseSIFTDescriptor()

    def add_sample_to_mem_bank(self, sample):
        sample = sample[2]
        dsift_feat = self.dsift(sample[:, 0, :, :].unsqueeze(dim=1)).detach()
        self.resize = torch.nn.AdaptiveAvgPool2d((28, 28))
        sift_depth_resized_maps = self.resize(self.average(dsift_feat))
        sift_patch = sift_depth_resized_maps.reshape(sift_depth_resized_maps.shape[1], -1).T
        self.patch_lib.append(sift_patch)

    def predict(self, sample, mask, label):
        sample = sample[2]
        feature_maps = self.dsift(sample[:, 0, :, :].unsqueeze(dim=1)).detach()
        self.resize = torch.nn.AdaptiveAvgPool2d((28, 28))
        depth_feature_maps_resized = self.resize(self.average(feature_maps))
        patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T

        self.compute_s_s_map(patch, depth_feature_maps_resized.shape[-2:], mask, label)
