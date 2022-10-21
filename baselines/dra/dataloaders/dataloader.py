from datasets import mvtecad
from torch.utils.data import DataLoader
from dataloaders.utlis import worker_init_fn_seed, BalancedBatchSampler

class initDataloader():

    @staticmethod
    def build(args, **kwargs):
        if args.dataset == "mvtecad":
            train_set = mvtecad.MVTecAD(args, train=True)
            test_set = mvtecad.MVTecAD(args, train=False)
            train_loader = DataLoader(train_set,
                                      worker_init_fn = worker_init_fn_seed,
                                      batch_sampler = BalancedBatchSampler(args, train_set),
                                      **kwargs)
            test_loader = DataLoader(test_set,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     worker_init_fn= worker_init_fn_seed,
                                     **kwargs)
            return train_loader, test_loader
        else:
            raise NotImplementedError