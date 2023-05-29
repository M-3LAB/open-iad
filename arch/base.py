import torch
from tools.utils import parse_device_list, seed_everything
from metric.cal_metric import CalMetric
from tools.record_helper import RecordHelper

class ModelBase():
    def __init__(self, config):
        self.config = config
        self.device, device_ids = parse_device_list(self.config['gpu_ids'], int(self.config['gpu_id']))
        seed_everything(self.config['seed'])

        # for training
        self.net = None
        self.optimizer = None
        self.scheduler = None

        # for test
        self.img_pred_list = [] # list<numpy>
        self.img_gt_list = [] # list<numpy>
        self.pixel_pred_list = [] # list<numpy(m,n)>
        self.pixel_gt_list = [] # list<numpy(m,n)>
        self.img_path_list = [] # list<str>

        # for computing result
        self.metric = CalMetric(self.config)
        # for recording result
        self.recorder = RecordHelper(self.config)

    def train_model(self, train_loader, task_id, inf=''):
        pass
    
    def train_epoch(self, train_loader, task_id, inf=''):
        pass

    def prediction(self, valid_loader, task_id=None):
        pass

    def visualization(self, vis_loader, task_id=None):
        self.clear_all_list()

        self.prediction(vis_loader, task_id)
        if len(self.pixel_gt_list)!=0 :
            self.recorder.record_images(self.img_pred_list, self.img_gt_list,
                                        self.pixel_pred_list, self.pixel_gt_list,
                                        self.img_path_list)
        
    def clear_all_list(self):
        self.img_pred_list = []
        self.img_gt_list = []
        self.pixel_pred_list = []
        self.pixel_gt_list = []
        self.img_path_list = []

    def cal_metric_all(self, task_id):
        # Logica AD Evaluation Needs Task ID and File Path
        return self.metric.cal_metric(self.img_pred_list, self.img_gt_list,
                                      self.pixel_pred_list, self.pixel_gt_list,
                                      self.img_path_list, task_id, self.config['file_path'])
    