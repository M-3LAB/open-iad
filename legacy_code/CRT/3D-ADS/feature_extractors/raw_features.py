from feature_extractors.features import Features


class RawFeatures(Features):
    def add_sample_to_mem_bank(self, sample):
        sample = sample[2]
        raw_depth_sample = sample[0, 0, :, :]
        raw_depth_sample_feature_maps = raw_depth_sample.reshape(28, 28, -1).permute(2, 0, 1).unsqueeze(dim=0)
        raw_depth_patch = raw_depth_sample_feature_maps.reshape(raw_depth_sample_feature_maps.shape[1], -1).T
        self.patch_lib.append(raw_depth_patch)

    def predict(self, sample, mask, label):
        sample = sample[2]
        sample = sample[0, 0, :, :]
        depth_feature_maps = sample.reshape(28, 28, -1).permute(2, 0, 1).unsqueeze(dim=0)
        patch = depth_feature_maps.reshape(depth_feature_maps.shape[1], -1).T


        self.compute_s_s_map(patch, depth_feature_maps.shape[-2:], mask, label)
