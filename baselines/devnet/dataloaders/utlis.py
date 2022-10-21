import numpy as np
from torch.utils.data import Sampler
from datasets.base_dataset import BaseADDataset


def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)


class BalancedBatchSampler(Sampler):
    def __init__(self,
                 cfg,
                 dataset: BaseADDataset):
        super(BalancedBatchSampler, self).__init__(dataset)
        self.cfg = cfg
        self.dataset = dataset

        self.normal_generator = self.random_generator(self.dataset.normal_idx)
        self.outlier_generator = self.random_generator(self.dataset.outlier_idx)
        if self.cfg.n_anomaly != 0:
            self.n_normal = self.cfg.batch_size // 2
            self.n_outlier = self.cfg.batch_size - self.n_normal
        else:
            self.n_normal = self.cfg.batch_size
            self.n_outlier = 0

    @staticmethod
    def random_generator(idx_list):
        while True:
            random_list = np.random.permutation(idx_list)
            for i in random_list:
                yield i

    def __len__(self):
        return self.cfg.steps_per_epoch
    
    def __iter__(self):
        for _ in range(self.cfg.steps_per_epoch):
            batch = []

            for _ in range(self.n_normal):
                batch.append(next(self.normal_generator))

            for _ in range(self.n_outlier):
                batch.append(next(self.outlier_generator))
            yield batch
