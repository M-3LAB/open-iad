import os
from rich import print
import cv2
import numpy as np

__all__ = ['RecordHelper']

class RecordHelper():
    def __init__(self, config):
        self.config = config

    def update(self, config):
        self.config = config
    
    def printer(self, info):
        print(info)

    def paradigm_name(self):
        if self.config['vanilla']:
            return 'vanilla'
        if self.config['semi']:
            return 'semi'
        if self.config['continual']:
            return 'continual'
        if self.config['fewshot']:
            return 'fewshot'
        if self.config['noisy']:
            return 'noisy'
        
        return 'unknown'

    def record_result(self, result):
        paradim = self.paradigm_name()
        save_dir = '{}/benchmark/{}/{}/{}/{}'.format(self.config['work_dir'], paradim, self.config['dataset'],
                                                                 self.config['model'], self.config['train_task_id_tmp'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = save_dir + '/result.txt'
        if paradim == 'vanilla':
            save_path = save_path
        if paradim == 'semi':
            save_path = '{}/result_{}.txt'.format(save_dir, self.config['semi_anomaly_num'])
        if paradim == 'fewshot':
            save_path = '{}/result_{}.txt'.format(save_dir, self.config['fewshot_exm'])
        if paradim == 'continual':
            save_path = '{}/result_{}.txt'.format(save_dir, self.config['valid_task_id_tmp'])
        if paradim == 'noisy':
            save_path = '{}/result_{}.txt'.format(save_dir, self.config['noisy_ratio'])
        
        with open(save_path, 'a') as f:
            print(result, file=f) 

    def record_images(self, img_pred_list, img_gt_list, pixel_pred_list, pixel_gt_list, img_path_list):
        paradim = self.paradigm_name()
        save_dir = '{}/benchmark/{}/{}/{}/{}'.format(self.config['work_dir'], paradim, self.config['dataset'],
                                                      self.config['model'], self.config['train_task_id_tmp'])
        
        if paradim == 'vanilla':
            save_dir = save_dir + '/vis'
        if paradim == 'semi':
            save_dir = '{}/vis_{}'.format(save_dir, self.config['semi_anomaly_num'])
        if paradim == 'fewshot':
            save_dir = '{}/vis_{}'.format(save_dir, self.config['fewshot_exm'])
        if paradim == 'continual':
            save_dir = '{}/vis_{}'.format(save_dir, self.config['valid_task_id_tmp'])
        if paradim == 'noisy':
            save_dir = '{}/vis_{}'.format(save_dir, self.config['noisy_ratio'])

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        img_shape = pixel_pred_list[0].shape
        for i in range(len(img_path_list)):
            path_dir = img_path_list[i][0].split('/')
            img_pred_path = '{}/{}_{}'.format(save_dir,path_dir[-2],path_dir[-1].replace('.','_heatmap.'))
            heatmap = self.cv2heatmap(pixel_pred_list[i]*255)
            cv2.imwrite(img_pred_path, heatmap)
            img_gt_path = '{}/{}_{}'.format(save_dir,path_dir[-2],path_dir[-1].replace('.','_gt.'))
            cv2.imwrite(img_gt_path, pixel_gt_list[i]*255)
            img_org_path = '{}/{}_{}'.format(save_dir,path_dir[-2],path_dir[-1].replace('.','_org.'))
            org_img = cv2.imread(img_path_list[i][0])
            org_img = cv2.resize(org_img,img_shape)
            cv2.imwrite(img_org_path, org_img)
            heatmap_on_img = self.heatmap_on_image(heatmap, org_img)
            img_heatmapOnImg_path = '{}/{}_{}'.format(save_dir,path_dir[-2],path_dir[-1].replace('.','_heatmapOnImg.'))
            cv2.imwrite(img_heatmapOnImg_path, heatmap_on_img)
    
    def heatmap_on_image(self, heatmap, image):
        if heatmap.shape != image.shape:
            heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
        out = np.float32(heatmap)/255 + np.float32(image)/255
        out = out / np.max(out)
        return np.uint8(255 * out)
    
    def cv2heatmap(self, gray):
        heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
        return heatmap
            