import torch
from feature_extractors.features import Features


class DepthInetFeatures(Features):

    def add_sample_to_mem_bank(self, sample):
        depth_inet_feature_maps = self(sample[2])
        if self.resize is None:
            depth_inet_largest_fmap_size = depth_inet_feature_maps[0].shape[-2:]
            self.resize = torch.nn.AdaptiveAvgPool2d(depth_inet_largest_fmap_size)
        depth_inet_resized_maps = [self.resize(self.average(fmap)) for fmap in depth_inet_feature_maps]
        depth_inet_patch = torch.cat(depth_inet_resized_maps, 1)
        depth_inet_patch = depth_inet_patch.reshape(depth_inet_patch.shape[1], -1).T
        self.patch_lib.append(depth_inet_patch)

    def predict(self, sample, mask, label):
        feature_maps = self(sample[2])
        resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
        patch = torch.cat(resized_maps, 1)
        patch = patch.reshape(patch.shape[1], -1).T

        self.compute_s_s_map(patch, feature_maps[0].shape[-2:], mask, label)
