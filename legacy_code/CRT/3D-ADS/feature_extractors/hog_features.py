from skimage.feature import hog
import torch
from feature_extractors.features import Features


class HoGFeatures(Features):
    def add_sample_to_mem_bank(self, sample):
        sample = sample[2]
        hog_feature = hog(sample[0, 0, :, :], orientations=8, pixels_per_cell=(8, 8),
                          cells_per_block=(1, 1), visualize=False, feature_vector=False)
        hog_feature = hog_feature.reshape(hog_feature.shape[0],
                                          hog_feature.shape[1],
                                          hog_feature.shape[2] *
                                          hog_feature.shape[3] *
                                          hog_feature.shape[4])
        hog_feature = torch.tensor(hog_feature.squeeze()).permute(2, 0, 1).unsqueeze(dim=0)
        hog_depth_patch = hog_feature.reshape(hog_feature.shape[1], -1).T
        self.patch_lib.append(hog_depth_patch)

    def predict(self, sample, mask, label):
        sample = sample[2]
        hog_features = hog(sample[0, 0, :, :], orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1),
                           visualize=False, feature_vector=False)
        hog_features = hog_features.reshape(hog_features.shape[0],
                                            hog_features.shape[1],
                                            hog_features.shape[2] * hog_features.shape[3] *
                                            hog_features.shape[4])
        depth_feature_maps_resized = torch.tensor(hog_features.squeeze()).permute(2, 0, 1).unsqueeze(dim=0)
        patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T
        self.compute_s_s_map(patch, depth_feature_maps_resized.shape[-2:], mask, label)
