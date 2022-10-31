from metrics.cal_metric import CalMetric
from tools.record_helper import RecordHelper

class ModelBase():
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net

        self.metric = CalMetric(self.config)
        self.img_pred_list = [] # list<numpy>
        self.img_gt_list = [] # list<numpy>
        self.pixel_pred_list = [] # list<numpy(m,n)>
        self.pixel_gt_list = [] # list<numpy(m,n)>
        self.img_path_list = [] # list<str>

        self.recorder = RecordHelper(self.config)

    def train_epoch(self, train_loader, task_id, inf=''):
        pass


    def prediction(self, valid_loader, task_id=None):
        pass

    def clear_all_list(self):
        self.img_pred_list = []
        self.img_gt_list = []
        self.pixel_pred_list = []
        self.pixel_gt_list = []
        self.img_path_list = []

    def cal_metric_all(self):
        return self.metric.cal_metric(self.img_pred_list, self.img_gt_list,
                                      self.pixel_pred_list, self.pixel_gt_list,
                                      self.img_path_list)