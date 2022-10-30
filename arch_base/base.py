from sklearn.metrics import roc_auc_score, average_precision_score
from metrics.mvtec3d.au_pro import calculate_au_pro
import numpy as np
import os
import cv2

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
        print(os.path.join(self.config['root_path'],self.config['data_path']))
         
    def train_epoch(self, train_loader, task_id, inf=''):
        pass


    def prediction(self, valid_loader, task_id=None):
        pass
    
    def cal_metric_all(self):
        # print(len(self.pixel_gt_list)) # n
        # print(len(self.pixel_pred_list)) # n
        # print(self.pixel_gt_list[0].shape) # 256
        # print(self.pixel_pred_list[0].shape) # 256
        if(self.config['dataset']!='mvtecloco'):
            pixel_pro, pro_curve = self.cal_metric_pixel_aupro()
            pixel_gt = np.array(self.pixel_gt_list).flatten()
            pixel_pred = np.array(self.pixel_pred_list).flatten()
            img_auroc = self.cal_metric_img_auroc()
            img_ap = self.cal_metric_img_ap()
            pixel_auroc = self.cal_metric_pixel_auroc(pixel_gt, pixel_pred)
            pixel_ap = self.cal_metric_pixel_ap(pixel_gt, pixel_pred)
        else:
            self.save_anomaly_map_tiff()
            pixel_auroc = 0
            img_auroc = 0
            pixel_ap = 0
            img_ap = 0
            pixel_pro = 0
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
    
    def save_anomaly_map_tiff(self):
        img_shape_list = {'breakfast_box': [1600,1280],
                     'juice_bottle': [800,1600],
                     'pushpins': [1700,1000],
                     'screw_bag': [1600,1100],
                     'splicing_connectors': [1700,850]}
        if self.config['vanilla']:
            train_type = 'vanilla'
        elif self.config['fewshot']:
            train_type = 'fewshot'
        elif self.config['continual']:
            train_type = 'continual'
        elif self.config['noisy']:
            train_type = 'noisy'
        elif self.config['fedrated']:
            train_type = 'fedrated'
        else:
            train_type = 'unknown'
        path_dir = self.img_path_list[0][0].split('/')
        img_shape = img_shape_list[path_dir[-4]]
        if not os.path.exists('./work_dir/'+train_type+'/'+self.config['dataset']+'/'+path_dir[-4]+'/'+'structural_anomalies'):
            os.makedirs('./work_dir/'+train_type+'/'+self.config['dataset']+'/'+path_dir[-4]+'/'+'structural_anomalies')
        if not os.path.exists('./work_dir/'+train_type+'/'+self.config['dataset']+'/'+path_dir[-4]+'/'+'logical_anomalies'):
            os.makedirs('./work_dir/'+train_type+'/'+self.config['dataset']+'/'+path_dir[-4]+'/'+'logical_anomalies')
        for i in range(len(self.img_path_list)):
            path_dir = self.img_path_list[i][0].split('/')
            anomaly_map = cv2.resize(self.pixel_pred_list[i],(img_shape[1],img_shape[0]))
            cv2.imwrite('./work_dir/'+train_type+'/'+self.config['dataset']+'/'+path_dir[-4]+'/'+path_dir[-2]+'/'+path_dir[-1].replace('png','tiff'),anomaly_map)
        
    #     def save_anomaly_map(anomaly_map, input_img, mask, file_path):
    #         if anomaly_map.shape != input_img.shape:
    #             anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))

    #         anomaly_map_norm = min_max_norm(anomaly_map) 
    #         heatmap = cv2heatmap(anomaly_map_norm*255)

    #         heatmap_on_img = heatmap_on_image(heatmap, input_img)
    #         #TODO: save problems
    #         create_folders(file_path)

    #         cv2.imwrite(os.path.join(file_path, 'input.jpg'), input_img)
    #         cv2.imwrite(os.path.join(file_path, 'heatmap.jpg'), heatmap)
    #         cv2.imwrite(os.path.join(file_path, 'heatmap_on_img.jpg'), heatmap_on_img)
    #         cv2.imwrite(os.path.join(file_path, 'mask.jpg'), mask)