from data_io.mvtec_ad import *

__all__ = ['CLData']
class CLData(object):
    
    def __init__(self, dataset, num_tasks=None):
        self.dataset = dataset

        self.data_list = []
        self.dataloader_list = []

        if self.dataset == 'mvtec2d':
            class_name = mvtec_2d_classes()
        elif self.dataset == 'mvtec3d':
            class_name = mvtec_3d_classes()

        self.num_tasks = len(class_name) if num_tasks is None else num_tasks            
        assert self.num_tasks <= len(class_name)

    def get_data(self):
        pass
         
    def get_dataloader(self): 
        pass