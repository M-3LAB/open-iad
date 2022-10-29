from sklearn.metrics import roc_auc_score, average_precision_score
from metrics.mvtec3d.au_pro import calculate_au_pro
import numpy as np
class ModelBase():
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
        self.img_pred_list = [] # list<numpy>
        self.img_gt_list = [] # list<numpy>
        self.pixel_pred_list = [] # list<numpy(m,n)>
        self.pixel_gt_list = [] # list<numpy(m,n)>
        self.img_path_list = [] # list<str>
        
        
    def train_epoch(self, train_loader, task_id, inf=''):
        pass


    def prediction(self, valid_loader, task_id=None):
        pass
    
    def cal_metric_all(self):
        # print(len(self.pixel_gt_list)) # n
        # print(len(self.pixel_pred_list)) # n
        # print(self.pixel_gt_list[0].shape) # 256
        # print(self.pixel_pred_list[0].shape) # 256
        pixel_pro, pro_curve = self.cal_metric_pixel_aupro()
        pixel_gt = np.array(self.pixel_gt_list).flatten()
        pixel_pred = np.array(self.pixel_pred_list).flatten()
        img_auroc = self.cal_metric_img_auroc()
        img_ap = self.cal_metric_img_ap()
        pixel_auroc = self.cal_metric_pixel_auroc(pixel_gt, pixel_pred)
        pixel_ap = self.cal_metric_pixel_ap(pixel_gt, pixel_pred)
        return pixel_auroc, img_auroc, pixel_ap, img_ap, pixel_pro
    
    def cal_metric_img_auroc(self):
        return roc_auc_score(self.img_gt_list, self.img_pred_list)
    
    def cal_metric_img_ap(self):
        return average_precision_score(self.img_gt_list, self.img_pred_list)
    
    def cal_metric_pixel_auroc(self, pixel_gt, pixel_pred):
        return roc_auc_score(pixel_gt, pixel_pred)
    
    def cal_metric_pixel_ap(self, pixel_gt, pixel_pred):
        return average_precision_score(pixel_gt, pixel_pred)
    
    def cal_metric_pixel_aupro(self):
        return calculate_au_pro(self.pixel_gt_list, self.pixel_pred_list)
    
        