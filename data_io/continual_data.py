
class CLData(object):
    
    def __init__(self, dataset, num_tasks):
        self.dataset = dataset
        self.num_tasks = num_tasks

        self.data_list = []
        self.dataloader_list = []

    def get_data(self):
        pass

    def get_dataloader(self): 
        pass