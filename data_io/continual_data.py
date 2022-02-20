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
        """
        Assign the number of classes for each task
        """
        num_classes_per_task = int(self.num_classes /  self.num_tasks)
        for i in range(num_classes_per_task):
            sub_class_name = self.class_name[self.num_tasks * i: self.num_tasks * (i + 1)]
            sub_dataset = data
         
    def get_dataloader(self): 
        for dataset in self.dataset_list:
            data_loader = DataLoader(dataset, self.config['batchsize'], 
                                      shuffle=True, 
                                      num_workers=self.config['num_workers'])
            
            self.dataloader_list.append(data_loader)