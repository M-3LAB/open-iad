import torch
import torch.nn as nn

__all__ = ['PaDim']

class PaDim():
    def __init__(self, config, train_loaders, valid_loaders, device, file_path, train_fewshot_loaders=None):
        
        self.config = config
        self.train_loader = train_loaders
        self.valid_loader = valid_loaders
        self.device = device
        self.file_path = file_path
        self.train_fewshot_loaders = train_fewshot_loaders

    def train_epoch(self, inf=''):
        for task_idx, train_loader in enumerate(self.train_loader):
            print('run task: {}'.format(task_idx))

            for i, batch in enumerate(train_loader):
                if self.config['debug'] and i > self.batch_limit:
                    break

                x, y, mask, task_id = batch
                
                pass


    def prediction(self):
        acc = 0

        for i, batch in enumerate(self.valid_loader):
            x, y, mask, task_id = batch
                
            pass

        return acc