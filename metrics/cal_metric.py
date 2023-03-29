import numpy as np
import cv2
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from metrics.mvtec3d.au_pro import calculate_au_pro
import json
import os
import numpy as np
from metrics.mvtec_loco_ad_evaluation.src.aggregation import MetricsAggregator, ThresholdMetrics
from metrics.mvtec_loco_ad_evaluation.src.image import GroundTruthMap, AnomalyMap, DefectsConfig
from metrics.mvtec_loco_ad_evaluation.src.util import get_auc_for_max_fpr,listdir, set_niceness, compute_classification_auc_roc
from metrics.mvtec_loco_ad_evaluation.evaluate_experiment import * 
from data_io.mvtecloco import mvtec_loco_classes
from data_io.miadloco import miad_loco_classes


__all__ = ['CalMetric']

class CalMetric():
    def __init__(self, config):
        self.config = config

        self.img_pred_list = [] # list<numpy>
        self.img_gt_list = [] # list<numpy>
        self.pixel_pred_list = [] # list<numpy(m,n)>
        self.pixel_gt_list = [] # list<numpy(m,n)>
        self.img_path_list = [] # list<str>
        
    def cal_metric(self, img_pred_list, img_gt_list, pixel_pred_list, pixel_gt_list, img_path_list, task_id):
        self.img_pred_list = img_pred_list
        self.img_gt_list = img_gt_list
        self.pixel_pred_list = pixel_pred_list
        self.pixel_gt_list = pixel_gt_list
        self.img_path_list = img_path_list

        pixel_auroc, img_auroc, pixel_ap, img_ap, pixel_pro = 0, 0, 0, 0, 0
        
        if(self.config['dataset']!='mvtecloco' and self.config['dataset'] != 'miadloco'):
            if(len(self.pixel_pred_list)!=0):
                pixel_pro, pro_curve = self.cal_pixel_aupro()
                self.pixel_gt_list = np.array(self.pixel_gt_list).flatten()
                self.pixel_pred_list = np.array(self.pixel_pred_list).flatten()
                pixel_auroc = self.cal_pixel_auroc()
                pixel_ap = self.cal_pixel_ap()
            if(len(self.img_pred_list)!=0):
                img_auroc = self.cal_img_auroc()
                img_ap = self.cal_img_ap()
        elif self.config['dataset'] == 'mvtecloco':
            if(len(self.pixel_pred_list)!=0):
                self.save_anomaly_map_tiff()
                pixel_pro = 1
            if(len(self.img_pred_list)!=0):
                self.cal_logical_metrics(task_id)
                #pixel_auroc, pixel_ap = self.cal_logical_img_auc()
                #img_auroc = self.cal_img_auroc()
                #img_ap = self.cal_img_ap()
        
        elif self.config['dataset'] == 'miadloco':
            #TODO 
            pass
        else:
            raise NotImplementedError('This type of dataset has not been implemented')
                
        return pixel_auroc, img_auroc, pixel_ap, img_ap, pixel_pro

    def cal_img_auroc(self):
        return roc_auc_score(self.img_gt_list, self.img_pred_list)
    
    def cal_img_ap(self):
        return average_precision_score(self.img_gt_list, self.img_pred_list)
    
    def cal_pixel_auroc(self):
        return roc_auc_score(self.pixel_gt_list, self.pixel_pred_list)
    
    def cal_pixel_ap(self):
        return average_precision_score(self.pixel_gt_list, self.pixel_pred_list)
    
    def cal_pixel_aupro(self):
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
        elif self.config['semi']:
            train_type = 'semi'
        elif self.config['fedrated']:
            train_type = 'fedrated'
        else:
            train_type = 'unknown'

        path_dir = self.img_path_list[0][0].split('/')
        print(f'path dir: {path_dir}')

        img_shape = img_shape_list[path_dir[-4]]
        print(f'img shape: {img_shape}')

        if train_type == 'continual':
            append_dir = '/'+str(self.config['train_task_id_tmp'])
        elif train_type == 'fewshot':
            append_dir = '/'+str(self.config['fewshot_exm'])
        elif train_type == 'noisy':
            append_dir = '/'+str(self.config['noisy_ratio'])
        elif train_type == 'semi':
            append_dir = '/'+str(self.config['semi_anomaly_num'])
        else:
            append_dir = ''
        if not os.path.exists('./work_dir/'+train_type+'/'+self.config['dataset']+'/'+self.config['model']+append_dir+'/'+path_dir[-4]+'/test/'+'structural_anomalies'):
            os.makedirs('./work_dir/'+train_type+'/'+self.config['dataset']+'/'+self.config['model']+append_dir+'/'+path_dir[-4]+'/test/'+'structural_anomalies')
        if not os.path.exists('./work_dir/'+train_type+'/'+self.config['dataset']+'/'+self.config['model']+append_dir+'/'+path_dir[-4]+'/test/'+'logical_anomalies'):
            os.makedirs('./work_dir/'+train_type+'/'+self.config['dataset']+'/'+self.config['model']+append_dir+'/'+path_dir[-4]+'/test/'+'logical_anomalies')
        if not os.path.exists('./work_dir/'+train_type+'/'+self.config['dataset']+'/'+self.config['model']+append_dir+'/'+path_dir[-4]+'/test/'+'good'):
            os.makedirs('./work_dir/'+train_type+'/'+self.config['dataset']+'/'+self.config['model']+append_dir+'/'+path_dir[-4]+'/test/'+'good')
        
        
        self.anomaly_map_dir = self.file_path+train_type+'/'+self.config['dataset']+'/'+self.config['model']+append_dir+'/'+path_dir[-4]
        self.json_dir = self.file_path+'logical_json'+train_type+'/'+self.config['dataset']+'/'+self.config['model']+append_dir+'/'+path_dir[-4]

        for i in range(len(self.img_path_list)):
            path_dir = self.img_path_list[i][0].split('/')
            anomaly_map = cv2.resize(self.pixel_pred_list[i],(img_shape[0],img_shape[1]))
            cv2.imwrite(self.file_path+train_type+'/'+self.config['dataset']+'/'+self.config['model']+append_dir+'/'+path_dir[-4]+'/test/'+path_dir[-2]+'/'+path_dir[-1].replace('png','tiff'),anomaly_map)
        
          
    def cal_logical_metrics(self, task_id):

        set_niceness(self.config['niceness'])
    
        object_name = mvtec_loco_classes[task_id]
        # Read the defects config file of the evaluated object.
        defects_config_path = os.path.join(
            self.config['data_path'], object_name, 'defects_config.json')
        with open(defects_config_path) as defects_config_file:
            defects_list = json.load(defects_config_file)
        defects_config = DefectsConfig.create_from_list(defects_list)

        # Read the ground truth maps and the anomaly maps.
        gt_dir = os.path.join(self.config['data_path'], object_name, 'ground_truth')
        anomaly_maps_test_dir = os.path.join(self.anomaly_maps_dir, object_name, 'test')
        gt_maps, anomaly_maps = read_maps(
        gt_dir=gt_dir,
        anomaly_maps_test_dir=anomaly_maps_test_dir,
        defects_config=defects_config)

        # Collect relevant metrics based on the ground truth and anomaly maps.
        metrics_aggregator = MetricsAggregator(
            gt_maps=gt_maps,
            anomaly_maps=anomaly_maps,
            parallel_workers=self.config['num_parallel_workers'],
            parallel_niceness=self.config['niceness'])

        metrics = metrics_aggregator.run(
            curve_max_distance=self.config['curve_max_distance'])
        
        # Fetch the anomaly localization results.
        localization_results = get_auc_spro_results(
            metrics=metrics,
            anomaly_maps_test_dir=anomaly_maps_test_dir)
        
        # Store the per-threshold metrics.
        results_per_threshold = {
            'thresholds': metrics.anomaly_thresholds.tolist(),
            'mean_spros': metrics.get_mean_spros().tolist(),
            'fp_rates': metrics.get_fp_rates().tolist(),
        }
        localization_results["per_threshold"] = results_per_threshold

        # Fetch the image-level anomaly detection results.
        classification_results = get_image_level_detection_metrics(
            gt_maps=gt_maps,
            anomaly_maps=anomaly_maps)
        
        # Create the dict to write to metrics.json.
        results = {
            'localization': localization_results, 
            'classification': classification_results
        }

        # Write the results to the output directory.
        if self.json_dir is not None:
            print(f'Writing results to {self.json_dir}') 
            os.makedirs(self.json_dir, exist_ok=True)
            # results_path = os.path.join(args.output_dir, 'metrics.json')
            results_path = os.path.join(self.json_dir, 'metrics_'+object_name+'.json')
            with open(results_path, 'w') as results_file:
                json.dump(results, results_file, indent=4, sort_keys=True)
        

    #def cal_logical_img_auc(self):
    #    img_pred_logical_list = []
    #    img_gt_logical_list = []
    #    img_pred_structural_list = []
    #    img_gt_structural_list = []
    #    for i in range(len(self.img_pred_list)):
    #        path_dir = self.img_path_list[i][0].split('/')
    #        if(path_dir[-2]=='good'):
    #            img_gt_logical_list.append(0)
    #            img_gt_structural_list.append(0)
    #            img_pred_logical_list.append(self.img_pred_list[i])
    #            img_pred_structural_list.append(self.img_pred_list[i])
    #        elif(path_dir[-2]=='logical_anomalies'):
    #            img_gt_logical_list.append(1)
    #            img_pred_logical_list.append(1)
    #        else:
    #            img_gt_structural_list.append(1)
    #            img_pred_structural_list.append(1)
    #            
    #    return roc_auc_score(img_gt_logical_list, img_pred_logical_list), roc_auc_score(img_gt_structural_list, img_pred_structural_list)