from data_io.mvtec_ad import *
from torch.utils.data import DataLoader

__all__ = ['CLData']

class CLData(object):
    
    def __init__(self, config, dataset_name, num_tasks=None):

        self.config = config

        self.dataset_name = dataset_name

        self.data_list = []
        self.dataloader_list = []

        if self.dataset_name == 'mvtec2d':
            self.class_name = mvtec_2d_classes()
        elif self.dataset_name == 'mvtec3d':
            self.class_name = mvtec_3d_classes()
        else:
            raise NotImplementedError('Not Implemented Yet')
        
        self.num_classes = len(self.class_name)

        self.num_tasks = self.num_classes if num_tasks is None else num_tasks            
        assert self.num_tasks <= len(self.class_name)

    def get_data(self):
        #for i in range():
        pass
         
    def get_dataloader(self): 
        for j in self.dataset_list:
            train_loader = DataLoader()