import torch





__all__ = ['PatchCore2D']


class PatchCore2D():
    def __init__(self, config, train_loader, valid_loader, device, file_path, batch_limit_weight=1.0):
        
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.file_path = file_path
        self.batch_limit_weight = batch_limit_weight
        self.batch_limit = 2

    def train_epoch(self, inf=''):
        for task_idx, train_loader in enumerate(self.train_loader):
            print('run task: {}'.format(task_idx))

            for i, batch in enumerate(train_loader):
                if self.config['debug'] and i > self.batch_limit:
                    break

                x, y, mask, task_id = batch
                
                pass


    def evaluation(self):
        acc = 0

        for i, batch in enumerate(self.valid_loader):
            x, y, mask, task_id = batch
                
            pass

        return acc