import yaml
import time
from tools.utils import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from data_io.fewshot import FewShot, extract_fewshot_data
from data_io.noisy import extract_noisy_data
from data_io.semi import extract_semi_data
from data_io.transfer import extract_transfer_data
from augmentation.domain_generalization import domain_gen
from augmentation.type import aug_type 

from configuration.device import assign_service
from configuration.registration import dataset_name, model_name

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

        if not (self.para_dict['vanilla'] or self.para_dict['transfer'] or self.para_dict['semi'] or self.para_dict['fewshot'] or self.para_dict['noisy'] or self.para_dict['continual']):
            raise ValueError('Please Assign Learning Paradigm, --vanilla, --transfer, --semi, --noisy, --fewshot, --continual')

    def preliminary(self):
        print('---------------------')
        print(self.args)
        print('---------------------')
        print(self.para_dict)
        print('---------------------')

        self.file_path = record_path(self.para_dict)
        self.para_dict['file_path'] = self.file_path

        if self.para_dict['save_log']:
            save_arg(self.para_dict, self.file_path)
            save_script(__file__, self.file_path)

        print('work dir: {}'.format(self.file_path))

    def load_data(self):
        dataset_package = __import__(dataset_name[self.para_dict['dataset']][0])
        dataset_module = getattr(dataset_package, dataset_name[self.para_dict['dataset']][1])
        dataset_class = getattr(dataset_module, dataset_name[self.para_dict['dataset']][2])

        train_data_transform =  aug_type(self.para_dict['train_aug_type'], self.para_dict)
        valid_data_transform =  aug_type(self.para_dict['valid_aug_type'], self.para_dict)

        self.train_dataset = dataset_class(data_path=self.para_dict['data_path'],
                                           learning_mode=self.para_dict['learning_mode'],
                                           phase='train',
                                           data_transform=train_data_transform,
                                           num_task=self.para_dict['num_task'])
        self.valid_dataset = dataset_class(data_path=self.para_dict['data_path'],
                                           learning_mode=self.para_dict['learning_mode'],
                                           phase='test',
                                           data_transform=valid_data_transform,
                                           num_task=self.para_dict['num_task'])

        self.refer_dataset = extract_fewshot_data(self.train_dataset, self.para_dict['ref_num'])
        self.vis_dataset = extract_fewshot_data(self.valid_dataset, self.para_dict['vis_num'])
        self.class_name = self.train_dataset.class_name

        if self.para_dict['fewshot']:
            self.train_fewshot_dataset = extract_fewshot_data(self.train_dataset, self.para_dict['fewshot_exm'])

        if self.para_dict['noisy']:
            self.train_noisy_dataset, self.valid_noisy_dataset, self.noisy_dataset = extract_noisy_data(self.train_dataset, 
                                                    self.valid_dataset, 
                                                    noisy_ratio=self.para_dict['noisy_ratio'], 
                                                    noisy_overlap=self.para_dict['noisy_overlap'])

        if self.para_dict['semi']:
            self.train_semi_dataset, self.valid_semi_dataset, self.semi_dataset = extract_semi_data(self.train_dataset, 
                                                    self.valid_dataset, 
                                                    anomaly_num=self.para_dict['semi_anomaly_num'], 
                                                    anomaly_overlap=self.para_dict['semi_overlap'])                                                    
        
        if self.para_dict['transfer']:
            self.train_transfer_dataset = extract_fewshot_data(self.train_dataset, self.para_dict['transfer_target_sample_num'])

        self.train_loaders, self.valid_loaders = [], []
        self.refer_loaders = []
        self.vis_loaders = []
        
        # vanilla training
        for i in range(self.para_dict['num_task']):
            train_task_data_list = self.train_dataset.sample_indices_in_task
            train_loader = DataLoader(self.train_dataset,
                                    batch_size=self.para_dict['train_batch_size'],
                                    num_workers=self.para_dict['num_workers'],
                                    sampler=SubsetRandomSampler(train_task_data_list[i]),
                                    drop_last=False)
            self.train_loaders.append(train_loader)

            valid_task_data_list = self.valid_dataset.sample_indices_in_task
            valid_loader = DataLoader(self.valid_dataset, 
                                    batch_size=self.para_dict['valid_batch_size'], 
                                    num_workers=self.para_dict['num_workers'],
                                    shuffle=False,
                                    sampler=SubsetRandomSampler(valid_task_data_list[i]),
                                    drop_last=False)
            self.valid_loaders.append(valid_loader)

            refer_task_data_list = self.refer_dataset.sample_indices_in_task
            refer_loader = DataLoader(self.train_dataset, 
                                    batch_size=self.para_dict['ref_num'], 
                                    num_workers=self.para_dict['num_workers'],
                                    sampler=SubsetRandomSampler(refer_task_data_list[i]),
                                    drop_last=False)
            self.refer_loaders.append(refer_loader) 

            vis_task_data_list = self.vis_dataset.sample_indices_in_task
            vis_loader = DataLoader(self.vis_dataset, 
                                    batch_size=self.para_dict['valid_batch_size'], 
                                    num_workers=self.para_dict['num_workers'],
                                    sampler=SubsetRandomSampler(vis_task_data_list[i]),
                                    drop_last=False)
            self.vis_loaders.append(vis_loader) 

        if self.para_dict['fewshot']:
            # capture few-shot images
            fewshot_images = []
            fewshot_task_data_list = self.train_fewshot_dataset.sample_indices_in_task
            for i in range(self.para_dict['num_task']):
                img_list = []
                for idx in fewshot_task_data_list[i]:
                    img_list.append(self.train_fewshot_dataset[idx])
                fewshot_images.append(img_list)
            # data augumentation
            if self.para_dict['fewshot_data_aug']:
                fewshot_images_dg = []
                for i in range(self.para_dict['num_task']):
                    data_gen_dataset = domain_gen(self.para_dict, fewshot_images[i])
                    fewshot_images_dg.append(data_gen_dataset)
                fewshot_images = fewshot_images_dg
            # back to normal training
            train_fewshot_loaders = []
            for i in range(self.para_dict['num_task']):
                fewshot_dg_datset = FewShot(fewshot_images[i])
                train_fewshot_loader = DataLoader(fewshot_dg_datset,
                                        batch_size=self.para_dict['train_batch_size'],
                                        num_workers=self.para_dict['num_workers'])
                train_fewshot_loaders.append(train_fewshot_loader)
            self.train_loaders = train_fewshot_loaders
                
        if self.para_dict['noisy']:
            train_task_data_list = self.train_noisy_dataset.sample_indices_in_task
            valid_task_data_list = self.valid_noisy_dataset.sample_indices_in_task 
            train_noisy_loaders, valid_noisy_loaders = [], []
            for i in range(self.para_dict['num_task']):
                train_noisy_loader = DataLoader(self.train_noisy_dataset,
                                        batch_size=self.para_dict['train_batch_size'],
                                        num_workers=self.para_dict['num_workers'],
                                        sampler=SubsetRandomSampler(train_task_data_list[i]))
                train_noisy_loaders.append(train_noisy_loader)

                valid_noisy_loader = DataLoader(self.valid_noisy_dataset, 
                                        batch_size=self.para_dict['valid_batch_size'], 
                                        num_workers=self.para_dict['num_workers'],
                                        shuffle=False,
                                        sampler=SubsetRandomSampler(valid_task_data_list[i]))
                valid_noisy_loaders.append(valid_noisy_loader)
            self.train_loaders = train_noisy_loaders
            self.valid_loaders = valid_noisy_loaders

        if self.para_dict['semi']:
            train_task_data_list = self.train_semi_dataset.sample_indices_in_task
            valid_task_data_list = self.valid_semi_dataset.sample_indices_in_task 
            train_semi_loaders, valid_semi_loaders = [], []
            for i in range(self.para_dict['num_task']):
                train_semi_loader = DataLoader(self.train_semi_dataset,
                                        batch_size=self.para_dict['train_batch_size'],
                                        num_workers=self.para_dict['num_workers'],
                                        sampler=SubsetRandomSampler(train_task_data_list[i]))
                train_semi_loaders.append(train_semi_loader)

                valid_semi_loader = DataLoader(self.valid_semi_dataset, 
                                        batch_size=self.para_dict['valid_batch_size'], 
                                        num_workers=self.para_dict['num_workers'],
                                        shuffle=False,
                                        sampler=SubsetRandomSampler(valid_task_data_list[i]))
                valid_semi_loaders.append(valid_semi_loader)
            self.train_loaders = train_semi_loaders
            self.valid_loaders = valid_semi_loaders

        if self.para_dict['transfer']:
            train_transfer_task_data_list = self.train_transfer_dataset.sample_indices_in_task
            self.train_transfer_loaders = []
            for i in range(self.para_dict['num_task']):
                train_transfer_loader = DataLoader(self.train_transfer_dataset,
                                        batch_size=self.para_dict['train_batch_size'],
                                        num_workers=self.para_dict['num_workers'],
                                        sampler=SubsetRandomSampler(train_transfer_task_data_list[i]),
                                        drop_last=False)
                self.train_transfer_loaders.append(train_transfer_loader)

        self.chosen_train_loaders, self.chosen_valid_loaders = [], []
        self.chosen_vis_loaders = []
        
        if self.para_dict['train_task_id'] == None or self.para_dict['valid_task_id'] == None:
            raise ValueError('Plase Assign Train Task Id!')

        if self.para_dict['vanilla'] or self.para_dict['semi'] or self.para_dict['fewshot'] or self.para_dict['noisy'] or self.para_dict['continual']:
            if self.para_dict['model'] == 'dra':
                for idx in self.para_dict['train_task_id']:
                    self.chosen_train_loaders.append([self.train_loaders[idx], self.refer_loaders[idx]])
                for idx in self.para_dict['valid_task_id']:
                    self.chosen_valid_loaders.append([self.valid_loaders[idx], self.refer_loaders[idx]])
                    self.chosen_vis_loaders.append([self.vis_loaders[idx], self.refer_loaders[idx]])
            else:
                for idx in self.para_dict['train_task_id']:
                    self.chosen_train_loaders.append(self.train_loaders[idx])
                for idx in self.para_dict['valid_task_id']:
                    self.chosen_valid_loaders.append(self.valid_loaders[idx])
                    self.chosen_vis_loaders.append(self.vis_loaders[idx])
        elif self.para_dict['transfer']:
            self.chosen_transfer_train_loaders, self.chosen_transfer_valid_loaders = [], []
            self.chosen_transfer_vis_loaders = []
            if self.para_dict['model'] == 'dra':
                for idx in self.para_dict['train_task_id']: # for step 1, train source task 
                    self.chosen_train_loaders.append([self.train_loaders[idx], self.refer_loaders[idx]])
                    self.chosen_valid_loaders.append([self.valid_loaders[idx], self.refer_loaders[idx]])
                    self.chosen_vis_loaders.append([self.vis_loaders[idx], self.refer_loaders[idx]])
                for idx in self.para_dict['valid_task_id']: # for step 2, train target task
                    self.chosen_transfer_train_loaders.append([self.train_transfer_loaders[idx], self.refer_loaders[idx]])
                    self.chosen_transfer_valid_loaders.append([self.valid_loaders[idx], self.refer_loaders[idx]])
                    self.chosen_transfer_vis_loaders.append([self.vis_loaders[idx], self.refer_loaders[idx]])
            else:
                for idx in self.para_dict['train_task_id']: # for step 1, train source task
                    self.chosen_train_loaders.append(self.train_loaders[idx])
                    self.chosen_valid_loaders.append(self.valid_loaders[idx])
                    self.chosen_vis_loaders.append(self.vis_loaders[idx])
                for idx in self.para_dict['valid_task_id']: # for step 2, train target task
                    self.chosen_transfer_train_loaders.append(self.train_transfer_loaders[idx])
                    self.chosen_transfer_valid_loaders.append(self.valid_loaders[idx])
                    self.chosen_transfer_vis_loaders.append(self.vis_loaders[idx])
        else:
            NotImplementedError('No this setting!')
    
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
        self.preliminary()
        self.load_data()
        self.init_model()
        print('---------------------')

        self.work_flow()
            
        print('work dir: {}'.format(self.file_path))
        with open('{}/log_finished.txt'.format(self.para_dict['work_dir']), 'a') as f:
            print('\n---> work dir {}'.format(self.file_path), file=f)
            print(self.args, file=f)
        print('---------------------')

