import torch
import yaml
from tools.utilize import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from data_io.mvtec2d import MVTec2D, MVTec2DFewShot
from data_io.mvtec3d import MVTec3D

from arch_base.patchcore2d import PatchCore2D
#from arch_base.pointcore3d import PointCore3D

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

        if self.para_dict['normal']:
            self.para_dict['num_task'] = 1

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

        if self.para_dict['dataset'] == 'mvtec2d':
            self.train_dataset = MVTec2D(data_path=self.para_dict['data_path'],
                                         learning_mode=self.para_dict['learning_mode'],
                                         phase='train',
                                         data_transform=mvtec2d_transform,
                                         num_task=self.para_dict['num_task'])
            self.valid_dataset = MVTec2D(data_path=self.para_dict['data_path'],
                                         learning_mode=self.para_dict['learning_mode'],
                                         phase='test',
                                         data_transform=mvtec2d_transform)
            if self.para_dict['fewshot']:
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
        elif self.para_dict['dataset'] == 'mtd':
            pass
        else:
            raise NotImplemented('Dataset Does Not Exist')

        self.train_loader = []
        task_data_list = self.train_dataset.sample_indices_in_task
        for i in range(self.para_dict['num_task']):
            train_loader = DataLoader(self.train_dataset,
                                      batch_size=self.para_dict['batch_size'],
                                      drop_last=True,
                                      num_workers=self.para_dict['num_workers'],
                                      sampler=SubsetRandomSampler(task_data_list[i]))
            self.train_loader.append(train_loader)

        if self.para_dict['fewshot']:
            self.train_fewshot_loader = []
            task_data_fewshot_list = self.train_fewshot_dataset.sample_indices_in_task
            for i in range(self.para_dict['num_task']):
                train_fewshot_loader = DataLoader(self.train_fewshot_dataset,
                                        batch_size=self.para_dict['batch_size'],
                                        drop_last=True,
                                        num_workers=self.para_dict['num_workers'],
                                        sampler=SubsetRandomSampler(task_data_fewshot_list[i]))
                self.train_fewshot_loader.append(train_fewshot_loader)

        self.valid_loader = DataLoader(self.valid_dataset, num_workers=self.para_dict['num_workers'],
                                 batch_size=self.para_dict['batch_size'], shuffle=False)
    def init_model(self):
        if self.para_dict['model'] == 'patchcore2d':
            self.trainer = PatchCore2D(self.para_dict, self.train_loader, self.valid_loader, self.device)
        else:
            raise ValueError('Model is invalid!')

        if self.para_dict['load_model']:
            self.load_model()
            print('load model: {}'.format(self.para_dict['load_model_dir']))

    def load_model(self):
        pass

    def save_model(self, psnr):
        pass

    def work_flow(self):
        self.trainer.train_epoch()
        acc = self.trainer.prediction()

        infor = '[Epoch {}/{}] acc: {:.4f}'.format(self.epoch+1, self.para_dict['num_epoch'], acc)

        print(infor)

        if self.para_dict['save_log']:
            save_log(infor, self.file_path, description='_clients')



    def run_work_flow(self):
        self.load_config()
        self.preliminary()
        self.load_data()
        self.init_model()
        print('---------------------')

        for epoch in range(self.para_dict['num_epoch']):
            self.epoch = epoch
            self.work_flow()
            
        print('work dir: {}'.format(self.file_path))
        with open('{}/log_finished.txt'.format(self.para_dict['work_dir']), 'a') as f:
            print('\n---> work dir {}'.format(self.file_path), file=f)
            print(self.args, file=f)
        print('---------------------')

