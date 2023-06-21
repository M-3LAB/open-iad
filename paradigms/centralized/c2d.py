import yaml
import time

from configuration.device import assign_service
from configuration.registration import setting_name, dataset_name, model_name
from data_io.data_holder import DataHolder
from tools.utils import *
from rich import print

import warnings
warnings.filterwarnings("ignore")

class CentralizedAD2D():
    def __init__(self, args):
        self.args = args
        
    def load_config(self):
        with open('./configuration/3_dataset_base/{}.yaml'.format(self.args.dataset), 'r') as f:
            config_dataset = yaml.load(f, Loader=yaml.SafeLoader)
        with open('./configuration/2_train_base/centralized_learning.yaml', 'r') as f:
            config_train = yaml.load(f, Loader=yaml.SafeLoader)
        with open('./configuration/1_model_base/{}.yaml'.format(self.args.model), 'r') as f:
            config_model = yaml.load(f, Loader=yaml.SafeLoader)
        
        config = override_config(config_dataset, config_train)
        config = override_config(config, config_model)
        self.para_dict = merge_config(config, self.args)
        self.args = extract_config(self.args)

        ip, root_path = assign_service(self.para_dict['server_moda'])
        print('local ip: {}, root_path: {}'.format(ip, root_path))

        self.para_dict['root_path'] = root_path
        self.para_dict['data_path'] = '{}{}'.format(root_path, self.para_dict['data_path'])
        self.para_dict['file_path'] = record_path(self.para_dict)

        if self.para_dict['save_log']:
            save_arg(self.para_dict, self.para_dict['file_path'])
            save_script(__file__, self.para_dict['file_path'])

        self.print_info()
        self.check_args()

    def check_args(self):
        n = 0
        for s in setting_name:
            n += self.para_dict[s]

        if n == 0:
            raise ValueError('Please Assign Learning Paradigm!')
        if n >= 2:
            raise ValueError('There Are Multiple Flags of Paradigm!')

    def print_info(self):
        print('---------------------')
        print(self.args)
        print('---------------------')
        print(self.para_dict)
        print('---------------------')

    def load_data(self):
        dataset_package = __import__(dataset_name[self.para_dict['dataset']][0])
        dataset_module = getattr(dataset_package, dataset_name[self.para_dict['dataset']][1])
        dataset_class = getattr(dataset_module, dataset_name[self.para_dict['dataset']][2])
        
        dataloader = DataHolder(dataset_class, self.para_dict)
        dataloader.create()

        self.chosen_train_loaders = dataloader.chosen_train_loaders
        self.chosen_valid_loaders = dataloader.chosen_valid_loaders
        self.chosen_vis_loaders = dataloader.chosen_vis_loaders

        self.chosen_transfer_train_loaders = dataloader.chosen_transfer_train_loaders
        self.chosen_transfer_valid_loaders = dataloader.chosen_transfer_valid_loaders
        self.chosen_transfer_vis_loaders = dataloader.chosen_transfer_vis_loaders

        self.class_name = dataloader.class_name
    
    def init_model(self):
        model_package = __import__(model_name[self.para_dict['model']][0])
        model_module = getattr(model_package, model_name[self.para_dict['model']][1])
        model_class = getattr(model_module, model_name[self.para_dict['model']][2])
        self.trainer = model_class(self.para_dict)
    
    def train_and_infer(self, train_loaders, valid_loaders, vis_loaders, train_task_ids, valid_task_ids):
        # train all task in one time
        for i, train_loader in enumerate(train_loaders):
            print('-> train ...')
            self.para_dict['train_task_id_tmp'] = train_task_ids[i]
            print('run task: {}, {}'.format(self.para_dict['train_task_id_tmp'], self.class_name[self.para_dict['train_task_id_tmp']]))
            self.trainer.train_model(train_loader, i)

            print('-> test ...')
            # test each task individually
            for j, (valid_loader, vis_loader) in enumerate(zip(valid_loaders, vis_loaders)):
                # for continual
                if j > i:
                    break
                self.para_dict['valid_task_id_tmp'] = valid_task_ids[j]
                print('run task: {}, {}'.format(self.para_dict['valid_task_id_tmp'], self.class_name[self.para_dict['valid_task_id_tmp']]))
                
                # calculate time 
                start_time = time.time()
                self.trainer.prediction(valid_loader, j)
                end_time = time.time()
                inference_speed = (end_time - start_time)/len(self.trainer.img_path_list)

                # calculate result
                pixel_auroc, img_auroc, pixel_ap, img_ap, pixel_aupro = self.trainer.cal_metric_all(
                    task_id=int(self.para_dict['train_task_id_tmp']))
                self.trainer.recorder.update(self.para_dict)

                paradim = self.trainer.recorder.paradigm_name()
                infor_basic = 'paradigm: {} dataset: {} model: {} train_task_id: {} valid_task_id: {}'.format(paradim, 
                    self.para_dict['dataset'], self.para_dict['model'], self.para_dict['train_task_id_tmp'], self.para_dict['valid_task_id_tmp'])
                
                # visualize result
                if self.para_dict['vis']:
                    print('-> visualize ...')
                    self.trainer.visualization(vis_loader, j)
                infor_result = 'pixel_auroc: {:.4f} img_auroc: {:.4f} pixel_ap: {:.4f} img_ap: {:.4f} pixel_aupro: {:.4f} inference speed: {:.4f}'.format(                                                                                  
                    pixel_auroc, img_auroc, pixel_ap, img_ap, pixel_aupro, inference_speed)
                self.trainer.recorder.printer('{} {}'.format(infor_basic, infor_result))

                # save result
                if self.para_dict['save_log']:
                    self.trainer.recorder.record_result(infor_result)

    def work_flow(self):
        if self.para_dict['vanilla'] or self.para_dict['semi'] or self.para_dict['fewshot'] or self.para_dict['noisy'] or self.para_dict['continual']:
            self.train_and_infer(self.chosen_train_loaders, self.chosen_valid_loaders, self.chosen_vis_loaders,
                                  self.para_dict['train_task_id'], self.para_dict['valid_task_id'])
        if self.para_dict['transfer']:
            self.train_and_infer(self.chosen_train_loaders, self.chosen_valid_loaders, self.chosen_vis_loaders,
                                  self.para_dict['train_task_id'], self.para_dict['train_task_id'])
            self.train_and_infer(self.chosen_transfer_train_loaders, self.chosen_transfer_valid_loaders, self.chosen_transfer_vis_loaders,
                                  self.para_dict['valid_task_id'], self.para_dict['valid_task_id'])

    def run_work_flow(self):
        self.load_config()
        
        self.load_data()
        self.init_model()

        self.work_flow()
