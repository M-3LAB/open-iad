import imp
from re import I
from xml.dom.minidom import DOMImplementation
import torch
import yaml
import argparse
import os
from tools.utils import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from data_io.fewshot import FewShot, extract_fewshot_data
from data_io.noisy import extract_noisy_data
from memory_augmentation.domain_generalization import domain_gen
from data_io.augmentation.type import aug_type 

from models.resnet.resnet import ResNetModel
from models.net_csflow.net_csflow import NetCSFlow
from models.vit.vit import ViT
from models.dream.draem import NetDRAEM
from arch_base.draem import weights_init
 
from models.optimizer import get_optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import models

# from arch_base.patchcore2d import PatchCore2D
# from arch_base.reverse import Reverse 
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

        ip, root_path = assign_service(self.para_dict['guoyang'])

        print('local ip: {}, root_path: {}'.format(ip, root_path))

        self.para_dict['root_path'] = root_path
        self.para_dict['data_path'] = '{}{}'.format(root_path, self.para_dict['data_path'])

        if not (self.para_dict['vanilla'] or self.para_dict['fewshot'] or self.para_dict['noisy'] or self.para_dict['continual']):
            raise ValueError('Please Assign Learning Paradigm, --vanilla, --noisy, --fewshot, --continual')

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
        dataset_name = {'mvtec2d': ('data_io.mvtec2d', 'mvtec2d', 'MVTec2D'),
                        'mvtec2df3d': ('data_io', 'mvtec2df3d', 'MVTec2DF3D'),
                        'mvtecloco': ('data_io', 'mvtecloco', 'MVTecLoco'),
                        'mpdd': ('data_io', 'mpdd', 'MPDD'),
                        'btad': ('data_io', 'btad', 'BTAD'),
                        'mtd': ('data_io', 'mtd', 'MTD'),
                        'mvtec3d': ('data_io', 'mvtec3d', 'MVTec3D'), }

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

        if self.para_dict['fewshot']:
            self.train_fewshot_dataset = extract_fewshot_data(self.train_dataset, self.para_dict['fewshot_exm'])

        if self.para_dict['noisy']:
            self.train_noisy_dataset, self.valid_noisy_dataset, self.noisy_dataset = extract_noisy_data(self.train_dataset, 
                                                    self.valid_dataset, 
                                                    noisy_ratio=self.para_dict['noisy_ratio'], 
                                                    noisy_overlap=self.para_dict['noisy_overlap'])
                                                    
        self.train_loaders, self.valid_loaders = [], []
        
        # vanilla training
        if self.para_dict['vanilla'] or self.para_dict['fewshot'] or self.para_dict['continual']:
            train_task_data_list = self.train_dataset.sample_indices_in_task
            valid_task_data_list = self.valid_dataset.sample_indices_in_task

            for i in range(self.para_dict['num_task']):
                train_loader = DataLoader(self.train_dataset,
                                        batch_size=self.para_dict['train_batch_size'],
                                        num_workers=self.para_dict['num_workers'],
                                        sampler=SubsetRandomSampler(train_task_data_list[i]))
                self.train_loaders.append(train_loader)

                valid_loader = DataLoader(self.valid_dataset, 
                                        num_workers=self.para_dict['num_workers'],
                                        batch_size=self.para_dict['valid_batch_size'], 
                                        shuffle=False,
                                        sampler=SubsetRandomSampler(valid_task_data_list[i]))
                self.valid_loaders.append(valid_loader)

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
            for i in range(self.para_dict['num_task']):
                train_loader = DataLoader(self.train_noisy_dataset,
                                        batch_size=self.para_dict['train_batch_size'],
                                        num_workers=self.para_dict['num_workers'],
                                        sampler=SubsetRandomSampler(train_task_data_list[i]))
                self.train_loaders.append(train_loader)

                valid_loader = DataLoader(self.valid_noisy_dataset, 
                                        num_workers=self.para_dict['num_workers'],
                                        batch_size=self.para_dict['valid_batch_size'], 
                                        shuffle=False,
                                        sampler=SubsetRandomSampler(valid_task_data_list[i]))
                self.valid_loaders.append(valid_loader)


        self.chosen_train_loaders, self.chosen_valid_loaders = [], []

        if self.para_dict['train_task_id'] == None or self.para_dict['valid_task_id'] == None:
            raise ValueError('Plase Assign Train Task Id!')

        for idx in self.para_dict['train_task_id']:
            self.chosen_train_loaders.append(self.train_loaders[idx])
        for idx in self.para_dict['valid_task_id']:
            self.chosen_valid_loaders.append(self.valid_loaders[idx])

    def init_model(self):
        self.net, self.optimizer, self.scheduler = None, None, None

        args = argparse.Namespace(**self.para_dict)
        if self.para_dict['net'] == 'resnet18': # patchcore
            self.net = models.resnet18(pretrained=True, progress=True)
        elif self.para_dict['net'] == 'wide_resnet50': # patchcore
            self.net = models.wide_resnet50_2(pretrained=True, progress=True)
        elif self.para_dict['net'] == 'net_csflow': # csflow
            self.net = NetCSFlow(args)
            self.optimizer = get_optimizer(args, self.net)
        elif self.para_dict['net'] == 'vit_b_16':
            self.net = ViT(num_classes=args._num_classes)
            if args._pretrained:
                checkpoint_path = './checkpoints/vit/vit_b_16.npz'
                if not os.path.exists(checkpoint_path):
                    os.system('wget https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz -O ./checkpoints/vit/vit_b_16.npz')
                self.net.load_pretrained(checkpoint_path)
            self.optimizer = get_optimizer(args, self.net)
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, args.num_epochs)
        elif self.para_dict['net'] == 'net_draem':
            self.net = NetDRAEM(args)
            self.net.apply(weights_init)
            self.optimizer = get_optimizer(args, self.net)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [args._num_epochs * 0.8, args._num_epochs * 0.9], gamma=0.2, last_epoch=-1)
        else:
            raise NotImplementedError('This Pretrained Model is Not Implemented Error')
        

        model_name = {'patchcore2d': ('arch_base.patchcore2d', 'patchcore2d', 'PatchCore2D'),
                      'csflow': ('arch_base.csflow', 'csflow', 'CSFlow'),
                      'dne': ('arch_base.dne', 'dne', 'DNE'),
                      'draem': ('arch_base.draem', 'draem', 'DRAEM'),
                     }

        model_package = __import__(model_name[self.para_dict['model']][0])
        model_module = getattr(model_package, model_name[self.para_dict['model']][1])
        model_class = getattr(model_module, model_name[self.para_dict['model']][2])

        self.trainer = model_class(self.para_dict, self.device, self.file_path, self.net, self.optimizer, self.scheduler)
       

    def work_flow(self):
        print('-> train ...')
        # train all task in one time
        train_loaders = self.chosen_train_loaders
        self.trainer.train_model(train_loaders)

        print('-> test ...')
        # test each task individually
        for task_id, valid_loader in enumerate(self.chosen_valid_loaders):
            pixel_auroc, img_auroc = self.trainer.prediction(valid_loader=valid_loader, task_id=task_id)

            infor = 'train_task_id: {} valid_task_id: {}'.format(self.para_dict['train_task_id'], self.para_dict['valid_task_id'][task_id])

            save_path = None 
            if self.para_dict['vanilla']:
                save_path = '{}/result_{}_vanilla.txt'.format(self.para_dict['work_dir'], self.para_dict['dataset']) 

            if self.para_dict['fewshot']:
                infor = '{} shot: {}'.format(infor, self.para_dict['fewshot_exm'])          
                save_path = '{}/result_{}_fewshot_{}.txt'.format(self.para_dict['work_dir'], self.para_dict['dataset'], self.para_dict['fewshot_exm']) 
                if self.para_dict['fewshot_data_aug']:
                    save_path = '{}/result_{}_fewshot_{}_da.txt'.format(self.para_dict['work_dir'], self.para_dict['dataset'], self.para_dict['fewshot_exm']) 
                if self.para_dict['fewshot_feat_aug']:
                    save_path = '{}/result_{}_fewshot_{}_fa.txt'.format(self.para_dict['work_dir'], self.para_dict['dataset'], self.para_dict['fewshot_exm']) 
                if self.para_dict['fewshot_data_aug'] and self.para_dict['fewshot_feat_aug']:
                    save_path = '{}/result_{}_fewshot_{}_ma.txt'.format(self.para_dict['work_dir'], self.para_dict['dataset'], self.para_dict['fewshot_exm']) 

            if self.para_dict['noisy']:
                infor = '{} noisy_ratio: {}'.format(infor, self.para_dict['noisy_ratio'])          
                save_path = '{}/result_{}_noisy.txt'.format(self.para_dict['work_dir'], self.para_dict['dataset']) 

            if self.para_dict['continual']:
                source_domain = ''
                for i in self.para_dict['train_task_id']:  
                    source_domain = source_domain + str(self.para_dict['train_task_id'][i])
                save_path = '{}/result_{}_continual_{}.txt'.format(self.para_dict['work_dir'], self.para_dict['dataset'], source_domain) 
            
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

        self.work_flow()
            
        print('work dir: {}'.format(self.file_path))
        with open('{}/log_finished.txt'.format(self.para_dict['work_dir']), 'a') as f:
            print('\n---> work dir {}'.format(self.file_path), file=f)
            print(self.args, file=f)
        print('---------------------')

