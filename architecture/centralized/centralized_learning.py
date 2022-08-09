from re import I
from xml.dom.minidom import DOMImplementation
import torch
import yaml
from tools.utilize import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from data_io.mvtec2d import MVTec2D, MVTec2DFewShot, FewShot
from data_io.mpdd import MPDD, MPDDFewShot, FewShot
from data_io.mvteclogical import MVTecLogical, MVTecLogicalFewShot, FewShot
from data_io.mvtec3d import MVTec3D
from memory_augmentation.domain_generalization import domain_gen

from arch_base.patchcore2d import PatchCore2D
from arch_base.reverse import Reverse 
#from arch_base.pointcore3d import PointCore3D
from configuration.architecture.config import assign_service

from rich import print

import warnings
warnings.filterwarnings("ignore")

class CentralizedTrain():
    def __init__(self, args):
        self.args = args

    def load_config(self):
        with open('./configuration/architecture/3_dataset_base/{}.yaml'.format(self.args.dataset), 'r') as f:
            config_model = yaml.load(f, Loader=yaml.SafeLoader)
        with open('./configuration/architecture/2_train_base/centralized_learning.yaml', 'r') as f:
            config_train = yaml.load(f, Loader=yaml.SafeLoader)
        with open('./configuration/architecture/1_model_base/{}.yaml'.format(self.args.model), 'r') as f:
            config_dataset = yaml.load(f, Loader=yaml.SafeLoader)

        config = override_config(config_model, config_train)
        config = override_config(config, config_dataset)
        self.para_dict = merge_config(config, self.args)
        self.args = extract_config(self.args)

        ip, root_path = assign_service()
        print('local ip: {}, root_path: {}'.format(ip, root_path))

        self.para_dict['root_path'] = root_path
        self.para_dict['data_path'] = '{}{}'.format(root_path, self.para_dict['data_path'])

    def preliminary(self):
        print('---------------------')
        print(self.args)
        print('---------------------')
        print(self.para_dict)
        print('---------------------')

        seed_everything(self.para_dict['seed'])

        device, device_ids = parse_device_list(self.para_dict['gpu_ids'], int(self.para_dict['gpu_id']))
        self.device = torch.device("cuda", device)

        self.file_path = record_path(self.para_dict)
        if self.para_dict['save_log']:
            save_arg(self.para_dict, self.file_path)
            save_script(__file__, self.file_path)

        print('work dir: {}'.format(self.file_path))

    def load_data(self):
        mvtec2d_transform = {'data_size':self.para_dict['data_size'],
                             'data_crop_size': self.para_dict['data_crop_size'],
                             'mask_size':self.para_dict['mask_size'],
                             'mask_crop_size': self.para_dict['mask_crop_size']}
        mvtec3d_transform = {'data_size':self.para_dict['data_size'],
                             'data_crop_size': self.para_dict['data_crop_size'],
                             'mask_size':self.para_dict['mask_size'],
                             'mask_crop_size': self.para_dict['mask_crop_size']}
        mpdd_transform = {'data_size':self.para_dict['data_size'],
                             'data_crop_size': self.para_dict['data_crop_size'],
                             'mask_size':self.para_dict['mask_size'],
                             'mask_crop_size': self.para_dict['mask_crop_size']}
        mvteclg_transform = {'data_size':self.para_dict['data_size'],
                             'data_crop_size': self.para_dict['data_crop_size'],
                             'mask_size':self.para_dict['mask_size'],
                             'mask_crop_size': self.para_dict['mask_crop_size']}
        if self.para_dict['dataset'] == 'mvtec2d':
            self.train_dataset = MVTec2D(data_path=self.para_dict['data_path'],
                                         learning_mode=self.para_dict['learning_mode'],
                                         phase='train',
                                         data_transform=mvtec2d_transform,
                                         num_task=self.para_dict['num_task'])
            self.valid_dataset = MVTec2D(data_path=self.para_dict['data_path'],
                                         learning_mode=self.para_dict['learning_mode'],
                                         phase='test',
                                         data_transform=mvtec2d_transform,
                                         num_task=self.para_dict['num_task'])
            if self.para_dict['fewshot'] or self.para_dict['fewshot_normal']:
                self.train_fewshot_dataset = MVTec2DFewShot(data_path=self.para_dict['data_path'],
                                                            learning_mode=self.para_dict['learning_mode'],
                                                            phase='train',
                                                            data_transform=mvtec2d_transform,
                                                            num_task=self.para_dict['num_task'],
                                                            fewshot_exm=self.para_dict['fewshot_exm'])
        elif self.para_dict['dataset'] == 'mvtec3d':
            self.train_dataset = MVTec3D(data_path=self.para_dict['data_path'],
                                         learning_mode=self.para_dict['learning_mode'],
                                         phase='train',
                                         data_transform=mvtec3d_transform,
                                         num_task=self.para_dict['num_task'])
            self.valid_dataset = MVTec3D(data_path=self.para_dict['data_path'],
                                         learning_mode=self.para_dict['learning_mode'],
                                         phase='test',
                                         data_transform=mvtec3d_transform)
        elif self.para_dict['dataset'] == 'mpdd':
            self.train_dataset = MPDD(data_path=self.para_dict['data_path'],
                                         learning_mode=self.para_dict['learning_mode'],
                                         phase='train',
                                         data_transform=mpdd_transform,
                                         num_task=self.para_dict['num_task'])
            self.valid_dataset = MPDD(data_path=self.para_dict['data_path'],
                                         learning_mode=self.para_dict['learning_mode'],
                                         phase='test',
                                         data_transform=mpdd_transform)
            if self.para_dict['fewshot'] or self.para_dict['fewshot_normal']:
                self.train_fewshot_dataset = MPDDFewShot(data_path=self.para_dict['data_path'],
                                                            learning_mode=self.para_dict['learning_mode'],
                                                            phase='train',
                                                            data_transform=mpdd_transform,
                                                            num_task=self.para_dict['num_task'],
                                                            fewshot_exm=self.para_dict['fewshot_exm'])
        elif self.para_dict['dataset'] == 'mvteclogical':
            self.train_dataset = MVTecLogical(data_path=self.para_dict['data_path'],
                                         learning_mode=self.para_dict['learning_mode'],
                                         phase='train',
                                         data_transform=mvteclg_transform,
                                         num_task=self.para_dict['num_task'])
            self.valid_dataset = MVTecLogical(data_path=self.para_dict['data_path'],
                                         learning_mode=self.para_dict['learning_mode'],
                                         phase='test',
                                         data_transform=mvteclg_transform)
            if self.para_dict['fewshot'] or self.para_dict['fewshot_normal']:
                self.train_fewshot_dataset = MVTecLogicalFewShot(data_path=self.para_dict['data_path'],
                                                            learning_mode=self.para_dict['learning_mode'],
                                                            phase='train',
                                                            data_transform=mvteclg_transform,
                                                            num_task=self.para_dict['num_task'],
                                                            fewshot_exm=self.para_dict['fewshot_exm'])
        else:
            raise NotImplemented('Dataset Does Not Exist')

        self.train_loaders = []
        self.valid_loaders = []

        train_task_data_list = self.train_dataset.sample_indices_in_task
        valid_task_data_list = self.valid_dataset.sample_indices_in_task

        if self.para_dict['model'] == 'patchcore2d':
            test_batch_size = 1
        else:
            test_batch_size = self.para_dict['batch_size']

        for i in range(self.para_dict['num_task']):
            train_loader = DataLoader(self.train_dataset,
                                      batch_size=self.para_dict['batch_size'],
                                      num_workers=self.para_dict['num_workers'],
                                      sampler=SubsetRandomSampler(train_task_data_list[i]))
            self.train_loaders.append(train_loader)

            valid_loader = DataLoader(self.valid_dataset, 
                                      num_workers=self.para_dict['num_workers'],
                                      batch_size=test_batch_size, 
                                      shuffle=False,
                                      sampler=SubsetRandomSampler(valid_task_data_list[i]))
            self.valid_loaders.append(valid_loader)

        if self.para_dict['fewshot'] or self.para_dict['fewshot_normal']:
            # capture few-shot images
            self.fewshot_images = []
            fewshot_task_data_list = self.train_fewshot_dataset.sample_indices_in_task
            for i in range(self.para_dict['num_task']):
                img_list = []
                for idx in fewshot_task_data_list[i]:
                    img_list.append(self.train_fewshot_dataset[idx])
                self.fewshot_images.append(img_list)
            # data augumentation
            if self.para_dict['data_aug']:
                self.fewshot_images_dg = []
                for i in range(self.para_dict['num_task']):
                    data_gen_dataset = domain_gen(self.para_dict, self.fewshot_images[i])
                    self.fewshot_images_dg.append(data_gen_dataset)
                self.fewshot_images = self.fewshot_images_dg
            # back to normal training
            self.train_fewshot_loaders = []
            for i in range(self.para_dict['num_task']):
                fewshot_dg_datset = FewShot(self.fewshot_images[i])
                train_fewshot_loader = DataLoader(fewshot_dg_datset,
                                        batch_size=self.para_dict['batch_size'],
                                        num_workers=self.para_dict['num_workers'])
                self.train_fewshot_loaders.append(train_fewshot_loader)

    def init_model(self):
        if self.para_dict['model'] == 'patchcore2d':
            if self.para_dict['fewshot'] or self.para_dict['fewshot_normal']:
                self.trainer = PatchCore2D(self.para_dict, self.train_loaders,  
                                           self.valid_loaders, self.device, self.file_path, 
                                           train_fewshot_loaders=self.train_fewshot_loaders)
            else:
                self.trainer = PatchCore2D(self.para_dict, self.train_loaders, self.valid_loaders, 
                                           self.device, self.file_path)
        elif self.para_dict['model'] == 'reverse':
            self.trainer = Reverse(self.para_dict, self.train_loaders, 
                                   self.valid_loaders, self.device, self.file_path) 
        else:
            raise ValueError('Model is invalid!')

       
    def work_flow(self):
        self.trainer.train_epoch()
        pixel_auroc, img_auroc = self.trainer.prediction()

        infor = 'train_task_id: {} test_task_id: {}'.format(self.para_dict['chosen_train_task_ids'], self.para_dict['chosen_test_task_id'])
        
        save_path = '{}/result_{}_normal.txt'.format(self.para_dict['work_dir'], self.para_dict['dataset']) 
        if self.para_dict['fewshot']:
            infor = '{} shot: {}'.format(infor, self.para_dict['fewshot_exm'])       
            save_path = '{}/result_{}_fewshot_{}.txt'.format(self.para_dict['work_dir'], self.para_dict['dataset'], self.para_dict['fewshot_exm']) 
        if self.para_dict['fewshot_normal']:
            infor = '{} shot: {}'.format(infor, self.para_dict['fewshot_exm'])       
            if self.para_dict['data_aug']:
                save_path = '{}/result_{}_fewshot_normal_{}_da.txt'.format(self.para_dict['work_dir'], self.para_dict['dataset'], self.para_dict['fewshot_exm']) 
            if self.para_dict['feat_aug']:
                save_path = '{}/result_{}_fewshot_normal_{}_fa.txt'.format(self.para_dict['work_dir'], self.para_dict['dataset'], self.para_dict['fewshot_exm']) 
            if self.para_dict['data_aug'] and self.para_dict['feat_aug']:
                save_path = '{}/result_{}_fewshot_normal_{}_ma.txt'.format(self.para_dict['work_dir'], self.para_dict['dataset'], self.para_dict['fewshot_exm']) 

        infor = '{} pixel_auroc: {:.4f} img_auroc: {:.4f}'.format(infor, pixel_auroc, img_auroc)

        print(infor)

        with open(save_path, 'a') as f:
            print(infor, file=f)

    def run_work_flow(self):
        self.load_config()
        self.preliminary()
        self.load_data()
        self.init_model()
        print('---------------------')

        for epoch in range(self.para_dict['num_epochs']):
            self.epoch = epoch
            self.work_flow()
            
        print('work dir: {}'.format(self.file_path))
        with open('{}/log_finished.txt'.format(self.para_dict['work_dir']), 'a') as f:
            print('\n---> work dir {}'.format(self.file_path), file=f)
            print(self.args, file=f)
        print('---------------------')

