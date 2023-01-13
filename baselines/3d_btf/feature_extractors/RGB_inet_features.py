import torch
from feature_extractors.features import Features




class RGBInetFeatures(Features):

    def add_sample_to_mem_bank(self, sample):
        rgb_feature_maps = self(sample[0])
        if self.resize is None:
            largest_fmap_size = rgb_feature_maps[0].shape[-2:]
            self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
        rgb_resized_maps = [self.resize(self.average(fmap)) for fmap in rgb_feature_maps]
        rgb_patch = torch.cat(rgb_resized_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        self.patch_lib.append(rgb_patch)

    def predict(self, sample, mask, label):
        feature_maps = self(sample[0])
        resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
        patch = torch.cat(resized_maps, 1)
        patch = patch.reshape(patch.shape[1], -1).T

        self.compute_s_s_map(patch, feature_maps[0].shape[-2:], mask, label)
