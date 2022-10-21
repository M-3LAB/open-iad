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

        self.normal_generator = self.randomGenerator(self.dataset.normal_idx)
        self.outlier_generator = self.randomGenerator(self.dataset.outlier_idx)
        if self.cfg.nAnomaly != 0:
            self.n_normal = 2 * self.cfg.batch_size // 3
            self.n_outlier = self.cfg.batch_size - self.n_normal
        else:
            self.n_normal = self.cfg.batch_size
            self.n_outlier = 0

    def randomGenerator(self, list):
        while True:
            random_list = np.random.permutation(list)
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