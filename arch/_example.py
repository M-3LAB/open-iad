from arch.base import ModelBase
from models._example.net_example import NetExample


__all__ = ['Example']

class ModelExample(ModelBase):
    def __init__(self, config):
        super(ModelExample, self).__init__(config)
        self.config = config
        self.net = NetExample(self.config)
    
    def train_model(self, train_loader, task_id, inf=''):
        pass
            
    def prediction(self, valid_loader, task_id=None):
        # implement these for test
        self.img_pred_list = [] # list<numpy>
        self.img_gt_list = [] # list<numpy>
        self.pixel_pred_list = [] # list<numpy(m,n)>
        self.pixel_gt_list = [] # list<numpy(m,n)>
        self.img_path_list = [] # list<str>