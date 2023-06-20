import os
import cv2
from tools.visualize import save_anomaly_map
from configuration.registration import setting_name
from rich import print

__all__ = ['RecordHelper']

class RecordHelper():
    def __init__(self, config):
        self.config = config

    def update(self, config):
        self.config = config
    
    def printer(self, info):
        print(info)

    def paradigm_name(self):
        for s in setting_name:
            if self.config[s]:
                return s

        print('Add new setting in record_helper.py!')
        return 'unknown'

    def record_result(self, result):
        paradim = self.paradigm_name()
        save_dir = '{}/benchmark/{}/{}/{}/task_{}'.format(self.config['work_dir'], paradim, self.config['dataset'],
                                                     self.config['model'], self.config['train_task_id_tmp'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = save_dir + '/result.txt'
        if paradim == 'vanilla':
            save_path = save_path
        if paradim == 'semi':
            save_path = '{}/result_{}_num.txt'.format(save_dir, self.config['semi_anomaly_num'])
        if paradim == 'fewshot':
            save_path = '{}/result_{}_{}_shot.txt'.format(save_dir, ''.join(self.config['fewshot_aug_type']), self.config['fewshot_exm'])
        if paradim == 'continual':
            save_path = '{}/result_{}_task.txt'.format(save_dir, self.config['valid_task_id_tmp'])
        if paradim == 'noisy':
            save_path = '{}/result_{}_ratio.txt'.format(save_dir, self.config['noisy_ratio'])
        if paradim == 'transfer':
            save_path = '{}/result_from_{}_to_{}.txt'.format(save_dir, self.config['train_task_id'][0], self.config['valid_task_id'][0]) 

        with open(save_path, 'a') as f:
            print(result, file=f) 

    def record_images(self, img_pred_list, img_gt_list, pixel_pred_list, pixel_gt_list, img_path_list):
        paradim = self.paradigm_name()
        save_dir = '{}/benchmark/{}/{}/{}/task_{}'.format(self.config['work_dir'], paradim, self.config['dataset'],
                                                      self.config['model'], self.config['train_task_id_tmp'])
        
        if paradim == 'vanilla':
            save_dir = save_dir + '/vis'
        if paradim == 'semi':
            save_dir = '{}/vis_{}_num'.format(save_dir, self.config['semi_anomaly_num'])
        if paradim == 'fewshot':
            save_dir = '{}/vis_{}_{}_shot'.format(save_dir, ''.join(self.config['fewshot_aug_type']), self.config['fewshot_exm'])
        if paradim == 'continual':
            save_dir = '{}/vis_{}_task'.format(save_dir, self.config['valid_task_id_tmp'])
        if paradim == 'noisy':
            save_dir = '{}/vis_{}_ratio'.format(save_dir, self.config['noisy_ratio'])
        if paradim == 'transfer':
            save_dir = '{}/vis_from_{}_to_{}'.format(save_dir, self.config['train_task_id'][0], self.config['valid_task_id'][0])

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        for i in range(len(img_path_list)):
            img_src = cv2.imread(img_path_list[i][0])
            img_src = cv2.resize(img_src, pixel_pred_list[0].shape)
            path_dir = img_path_list[i][0].split('/')
            save_path = '{}/{}_{}'.format(save_dir, path_dir[-2], path_dir[-1][:-4])

            save_anomaly_map(anomaly_map=pixel_pred_list[i], input_img=img_src, mask=pixel_gt_list[i], file_path=save_path)
