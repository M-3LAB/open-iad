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
from data_io.semi import extract_semi_data
from memory_augmentation.domain_generalization import domain_gen
from data_io.augmentation.type import aug_type 

from models.net_csflow.net_csflow import NetCSFlow
from models.vit.vit import ViT
from models.dream.draem import NetDRAEM
from models.dra.dra_resnet18 import DraResNet18
from models.devnet.devnet_resnet18 import DevNetResNet18
from models.igd.net_igd import NetIGD
from models.reverse.net_reverse import NetReverse
from models.fastflow.net import NetFastFlow
from models.cfa.net_cfa import NetCFA
 
from optimizer.optimizer import get_optimizer
from models.favae.net_favae import NetFAVAE
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import models

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

        if not (self.para_dict['vanilla'] or self.para_dict['semi'] or self.para_dict['fewshot'] or self.para_dict['noisy'] or self.para_dict['continual']):
            raise ValueError('Please Assign Learning Paradigm, --vanilla, --semi, --noisy, --fewshot, --continual')

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
                        'mvtecloco': ('data_io.mvtecloco', 'mvtecloco', 'MVTecLoco'),
                        'mpdd': ('data_io', 'mpdd', 'MPDD'),
                        'btad': ('data_io', 'btad', 'BTAD'),
                        'mtd': ('data_io', 'mtd', 'MTD'),
                        'mvtec3d': ('data_io', 'mvtec3d', 'MVTec3D'), 
                        }

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

        self.train_semi_dataset, self.valid_semi_dataset, self.semi_dataset = extract_semi_data(self.train_dataset, 
                                                self.valid_dataset, 
                                                anomaly_num=self.para_dict['semi_anomaly_num'], 
                                                anomaly_overlap=self.para_dict['semi_overlap'])                                                    

        if self.para_dict['fewshot']:
            self.train_fewshot_dataset = extract_fewshot_data(self.train_dataset, self.para_dict['fewshot_exm'])

        if self.para_dict['noisy']:
            self.train_noisy_dataset, self.valid_noisy_dataset, self.noisy_dataset = extract_noisy_data(self.train_dataset, 
                                                    self.valid_dataset, 
                                                    noisy_ratio=self.para_dict['noisy_ratio'], 
                                                    noisy_overlap=self.para_dict['noisy_overlap'])

        if self.para_dict['model'] == 'devnet':
            self.train_dataset = self.train_semi_dataset

        self.train_loaders, self.valid_loaders = [], []
        self.refer_loaders = []
        
        # vanilla training
        train_task_data_list = self.train_dataset.sample_indices_in_task
        valid_task_data_list = self.valid_dataset.sample_indices_in_task
        semi_task_data_list = self.semi_dataset.sample_indices_in_task

        for i in range(self.para_dict['num_task']):
            train_loader = DataLoader(self.train_dataset,
                                    batch_size=self.para_dict['train_batch_size'],
                                    num_workers=self.para_dict['num_workers'],
                                    sampler=SubsetRandomSampler(train_task_data_list[i]),
                                    drop_last=True)
            self.train_loaders.append(train_loader)

            valid_loader = DataLoader(self.valid_dataset, 
                                    num_workers=self.para_dict['num_workers'],
                                    batch_size=self.para_dict['valid_batch_size'], 
                                    shuffle=False,
                                    sampler=SubsetRandomSampler(valid_task_data_list[i]),
                                    drop_last=True)
            self.valid_loaders.append(valid_loader)

            semi_loader = DataLoader(self.semi_dataset, 
                                batch_size=self.para_dict['semi_anomaly_num'], 
                                num_workers=self.para_dict['num_workers'],
                                sampler=SubsetRandomSampler(semi_task_data_list[i]),
                                drop_last=True)
            self.refer_loaders.append(semi_loader) 

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

        if self.para_dict['model'] == 'dra':
            for idx in self.para_dict['train_task_id']:
                self.chosen_train_loaders.append([self.train_loaders[idx], self.refer_loaders[idx]])
            for idx in self.para_dict['valid_task_id']:
                self.chosen_valid_loaders.append([self.valid_loaders[idx], self.refer_loaders[idx]])
        else:
            for idx in self.para_dict['train_task_id']:
                self.chosen_train_loaders.append(self.train_loaders[idx])
            for idx in self.para_dict['valid_task_id']:
                self.chosen_valid_loaders.append(self.valid_loaders[idx])

    def init_model(self):
        self.net, self.optimizer, self.scheduler = None, None, None

        if self.para_dict['net'] == 'resnet18': 
            self.net = models.resnet18(pretrained=True, progress=True)
        if self.para_dict['net'] == 'wide_resnet50':
            self.net = models.wide_resnet50_2(pretrained=True, progress=True)

        args = argparse.Namespace(**self.para_dict)
        if self.para_dict['net'] == 'net_csflow': 
            self.net = NetCSFlow(args)
            self.optimizer = get_optimizer(args, self.net.density_estimator.parameters())
        if self.para_dict['net'] == 'vit_b_16':
            self.net = ViT(num_classes=args._num_classes, pretrained=args._pretrained, checkpoint_path='./checkpoints/vit/vit_b_16.npz')
            self.optimizer = get_optimizer(args, self.net.parameters())
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, args.num_epochs)
        if self.para_dict['net'] == 'net_draem':
            self.net = NetDRAEM(args)
            self.optimizer = get_optimizer(args, self.net.parameters())
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [args.num_epochs * 0.8, args.num_epochs * 0.9], gamma=args._gamma, last_epoch=-1)
        if self.para_dict['net'] == 'net_igd':
            self.net = NetIGD(args)
            self.optimizer_g = get_optimizer(args, self.net.g.parameters())
            self.optimizer_d = get_optimizer(args, self.net.d.parameters())
            self.scheduler_g = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_g, [args.num_epochs * 0.8, args.num_epochs * 0.9], gamma=args._gamma, last_epoch=-1)
            self.scheduler_d = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_d, [args.num_epochs * 0.8, args.num_epochs * 0.9], gamma=args._gamma, last_epoch=-1)
            self.optimizer = [self.optimizer_g, self.optimizer_d]
            self.scheduler = [self.scheduler_g, self.scheduler_d]
        if self.para_dict['net'] == 'net_dra':
            self.net = DraResNet18()
            self.optimizer = get_optimizer(args, self.net.parameters())
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args._step_size, gamma=args._gamma)
        if self.para_dict['net'] == 'net_devnet':
            self.net = DevNetResNet18()
            self.optimizer = get_optimizer(args, self.net.parameters())
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args._step_size, gamma=args._gamma)       
        if self.para_dict['net'] == 'net_favae':
            self.net = NetFAVAE() 
            self.optimizer = get_optimizer(args, self.net.parameters())
            self.scheduler = None
        if self.para_dict['net'] == 'net_reverse':
            self.net = NetReverse(args) 
            self.optimizer = get_optimizer(args, list(self.net.decoder.parameters()) + list(self.net.bn.parameters()))
        if self.para_dict['model'] == 'fastflow':
            self.net = NetFastFlow(self.para_dict) 
            self.optimizer = get_optimizer(args, self.net.parameters())
            self.scheduler = None
        if self.para_dict['model'] == 'stpm':
            self.optimizer = get_optimizer(args, self.net.parameters())
        if self.para_dict['net'] == 'net_cfa':
            self.net = NetCFA(args)
            self.optimizer = None
            self.scheduler = None
        if self.para_dict['model'] == 'cutpaste':
            self.optimizer = get_optimizer(args, self.net.parameters())
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, args.num_epochs) 

        model_name = {'patchcore': ('arch_base.patchcore', 'patchcore', 'PatchCore'),
                      'padim': ('arch_base.padim', 'padim', 'PaDim'),
                      'csflow': ('arch_base.csflow', 'csflow', 'CSFlow'),
                      'dne': ('arch_base.dne', 'dne', 'DNE'),
                      'draem': ('arch_base.draem', 'draem', 'DRAEM'),
                      'igd': ('arch_base.igd', 'igd', 'IGD'),
                      'dra': ('arch_base.dra', 'dra', 'DRA'),
                      'devnet': ('arch_base.devnet', 'devnet', 'DevNet'),
                      'favae': ('arch_base.favae', 'favae', 'FAVAE'),
                      'fastflow': ('arch_base.fastflow', 'fastflow', 'FastFlow'),
                      'cfa': ('arch_base.cfa', 'cfa', 'CFA'),
                      'reverse': ('arch_base.reverse', 'reverse', 'REVERSE'),
                      'spade': ('arch_base.spade', 'spade', 'SPADE'),
                      'stpm': ('arch_base.stpm', 'stpm', 'STPM'),
                      'cutpaste': ('arch_base.cutpaste', 'cutpaste', 'CutPaste')
                     }

        model_package = __import__(model_name[self.para_dict['model']][0])
        model_module = getattr(model_package, model_name[self.para_dict['model']][1])
        model_class = getattr(model_module, model_name[self.para_dict['model']][2])

        self.trainer = model_class(self.para_dict, self.device, self.file_path, self.net, self.optimizer, self.scheduler)
       

    def work_flow(self):
        print('-> train ...')
        # train all task in one time
        for task_idx, train_loader in enumerate(self.chosen_train_loaders):
            print('run task: {}'.format(self.para_dict['train_task_id'][task_idx]))
            self.trainer.train_model(train_loader, task_idx)

        print('-> test ...')
        # test each task individually
        for task_id, valid_loader in enumerate(self.chosen_valid_loaders):
            self.trainer.prediction(valid_loader, task_id=task_id)
            pixel_auroc, img_auroc, pixel_ap, img_ap, pixel_aupro = self.trainer.cal_metric_all()

            infor = 'dataset_name: {} model_name: {} train_task_id: {} valid_task_id: {}'.format(\
                self.para_dict['dataset'], self.para_dict['model'], self.para_dict['train_task_id'], self.para_dict['valid_task_id'][task_id])

            save_path = None 
            if self.para_dict['vanilla']:
                save_path = '{}/result_{}_vanilla.txt'.format(self.para_dict['work_dir'], self.para_dict['dataset']) 

            if self.para_dict['semi']:
                save_path = '{}/result_{}_semi.txt'.format(self.para_dict['work_dir'], self.para_dict['dataset']) 

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
            
            infor = '{} pixel_auroc: {:.4f} img_auroc: {:.4f} pixel_ap: {:.4f} img_ap: {:.4f} pixel_aupro: {:.4f}'.format(\
                infor, pixel_auroc, img_auroc, pixel_ap, img_ap, pixel_aupro)
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

