import numpy as np
from arch_base.base import ModelBase
from models.patchcore.patchcore import PatchCore as patchcore_official
from models.patchcore import common
from models.patchcore import sampler

__all__ = ['PatchCore']

class PatchCore(ModelBase):
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        super(PatchCore, self).__init__(config, device, file_path, net, optimizer, scheduler)
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
        self.sampler = self.get_sampler(self.config['_sampler_name'])
        self.nn_method = common.FaissNN(self.config['_faiss_on_gpu'], self.config['_faiss_num_workers'])

        self.patchcore_instance = patchcore_official(self.device)
        self.patchcore_instance.load(
            backbone=self.net,
            layers_to_extract_from=self.config['_layers_to_extract_from'],
            device=self.device,
            input_shape=self.config['_input_shape'],
            pretrain_embed_dimension=self.config['_pretrain_embed_dimension'],
            target_embed_dimension=self.config['_target_embed_dimension'],
            patchsize=self.config['_patch_size'],
            featuresampler=self.sampler,
            anomaly_scorer_num_nn=self.config['_anomaly_scorer_num_nn'],
            nn_method=self.nn_method,
        )

    def get_sampler(self, name):
        if name == 'identity':
            return sampler.IdentitySampler()
        elif name == 'greedy_coreset':
            return sampler.GreedyCoresetSampler(self.config['sampler_percentage'], self.device)
        elif name == 'approx_greedy_coreset':
            return sampler.ApproximateGreedyCoresetSampler(self.config['sampler_percentage'], self.device)
        else:
            raise ValueError('No This Sampler: {}'.format(name))

    def train_model(self, train_loader, task_id, inf=''):
        self.patchcore_instance.eval()
        self.patchcore_instance.fit(train_loader)

    def prediction(self, valid_loader, task_id=None):
        self.patchcore_instance.eval()
        self.clear_all_list()

        scores, segmentations, labels_gt, masks_gt = self.patchcore_instance.predict(valid_loader)

        scores = np.array(scores)
        min_scores = scores.min(axis=-1).reshape(-1, 1)
        max_scores = scores.max(axis=-1).reshape(-1, 1)
        scores = (scores - min_scores) / (max_scores - min_scores)
        scores = np.mean(scores, axis=0)

        segmentations = np.array(segmentations)
        min_scores = segmentations.reshape(len(segmentations), -1).min(axis=-1).reshape(-1, 1, 1, 1)
        max_scores = segmentations.reshape(len(segmentations), -1).max(axis=-1).reshape(-1, 1, 1, 1)
        segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        segmentations = np.mean(segmentations, axis=0)
        segmentations[segmentations >= 0.5] = 1
        segmentations[segmentations < 0.5] = 0
        segmentations = segmentations.astype(int) 
        masks_gt = np.array(masks_gt).squeeze().astype(int)

        self.pixel_gt_list = [mask for mask in masks_gt]
        self.pixel_pred_list = [seg for seg in segmentations]
        self.img_gt_list = labels_gt
        self.img_pred_list = scores

        for batch in valid_loader:
            self.img_path_list.append(batch['img_src'])


        
      

