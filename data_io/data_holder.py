from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from data_io.fewshot import FewShot, extract_fewshot_data
from data_io.noisy import extract_noisy_data
from data_io.semi import extract_semi_data
from augmentation.domain_gen import domain_gen
from augmentation.type import aug_type 

class DataHolder(object):
    def __init__(self, dataset_class, config):
        self.config = config
        self.dataset_class = dataset_class

        train_data_transform =  aug_type(self.config['train_aug_type'], self.config)
        valid_data_transform =  aug_type(self.config['valid_aug_type'], self.config)

        self.train_dataset = dataset_class(data_path=self.config['data_path'],
                                           learning_mode=self.config['learning_mode'],
                                           phase='train',
                                           data_transform=train_data_transform,
                                           num_task=self.config['num_task'])
        self.valid_dataset = dataset_class(data_path=self.config['data_path'],
                                           learning_mode=self.config['learning_mode'],
                                           phase='test',
                                           data_transform=valid_data_transform,
                                           num_task=self.config['num_task'])

        self.refer_dataset = extract_fewshot_data(self.train_dataset, self.config['ref_num'])
        self.vis_dataset = extract_fewshot_data(self.valid_dataset, self.config['vis_num'])
        self.class_name = self.train_dataset.class_name

        self.train_loaders, self.valid_loaders = [], []
        self.refer_loaders, self.vis_loaders = [], []
        self.chosen_train_loaders, self.chosen_valid_loaders = [], []
        self.chosen_vis_loaders = []
        # only for transfer
        self.chosen_transfer_train_loaders, self.chosen_transfer_valid_loaders = [], []
        self.chosen_transfer_vis_loaders = []
        
        # vanilla training
        for i in range(self.config['num_task']):
            train_task_data_list = self.train_dataset.sample_indices_in_task
            train_loader = DataLoader(self.train_dataset,
                                    batch_size=self.config['train_batch_size'],
                                    num_workers=self.config['num_workers'],
                                    sampler=SubsetRandomSampler(train_task_data_list[i]),
                                    drop_last=False)
            self.train_loaders.append(train_loader)

            valid_task_data_list = self.valid_dataset.sample_indices_in_task
            valid_loader = DataLoader(self.valid_dataset, 
                                    batch_size=self.config['valid_batch_size'], 
                                    num_workers=self.config['num_workers'],
                                    shuffle=False,
                                    sampler=SubsetRandomSampler(valid_task_data_list[i]),
                                    drop_last=False)
            self.valid_loaders.append(valid_loader)

            refer_task_data_list = self.refer_dataset.sample_indices_in_task
            refer_loader = DataLoader(self.train_dataset, 
                                    batch_size=self.config['ref_num'], 
                                    num_workers=self.config['num_workers'],
                                    sampler=SubsetRandomSampler(refer_task_data_list[i]),
                                    drop_last=False)
            self.refer_loaders.append(refer_loader) 

            vis_task_data_list = self.vis_dataset.sample_indices_in_task
            vis_loader = DataLoader(self.vis_dataset, 
                                    batch_size=self.config['valid_batch_size'], 
                                    num_workers=self.config['num_workers'],
                                    sampler=SubsetRandomSampler(vis_task_data_list[i]),
                                    drop_last=False)
            self.vis_loaders.append(vis_loader) 

    def create(self):
        if self.config['vanilla']:
            self.create_vanilla()

        if self.config['fewshot']:
            self.create_fewshot()

        if self.config['noisy']:
            self.create_noisy()

        if self.config['semi']:
            self.create_semi()

        if self.config['continual']:
            self.create_continual()

        if self.config['transfer']:
            self.create_transfer()

    def create_vanilla(self):
        self.create_chosen_dataloaders()

    def create_fewshot(self):
        self.train_fewshot_dataset = extract_fewshot_data(self.train_dataset, self.config['fewshot_exm'])
        # capture few-shot images
        fewshot_images = []
        fewshot_task_data_list = self.train_fewshot_dataset.sample_indices_in_task
        for i in range(self.config['num_task']):
            img_list = []
            for idx in fewshot_task_data_list[i]:
                img_list.append(self.train_fewshot_dataset[idx])
            fewshot_images.append(img_list)
        # data augumentation
        if self.config['fewshot_data_aug']:
            fewshot_images_dg = []
            for i in range(self.config['num_task']):
                data_gen_dataset = domain_gen(self.config, fewshot_images[i])
                fewshot_images_dg.append(data_gen_dataset)
            fewshot_images = fewshot_images_dg
        # back to normal training
        train_fewshot_loaders = []
        for i in range(self.config['num_task']):
            fewshot_dg_datset = FewShot(fewshot_images[i])
            train_fewshot_loader = DataLoader(fewshot_dg_datset,
                                    batch_size=self.config['train_batch_size'],
                                    num_workers=self.config['num_workers'])
            train_fewshot_loaders.append(train_fewshot_loader)
        self.train_loaders = train_fewshot_loaders
        self.create_chosen_dataloaders()

    def create_noisy(self):
        self.train_noisy_dataset, self.valid_noisy_dataset, self.noisy_dataset = extract_noisy_data(self.train_dataset, 
                                                self.valid_dataset, 
                                                noisy_ratio=self.config['noisy_ratio'], 
                                                noisy_overlap=self.config['noisy_overlap'])
        train_task_data_list = self.train_noisy_dataset.sample_indices_in_task
        valid_task_data_list = self.valid_noisy_dataset.sample_indices_in_task 
        train_noisy_loaders, valid_noisy_loaders = [], []
        for i in range(self.config['num_task']):
            train_noisy_loader = DataLoader(self.train_noisy_dataset,
                                    batch_size=self.config['train_batch_size'],
                                    num_workers=self.config['num_workers'],
                                    sampler=SubsetRandomSampler(train_task_data_list[i]))
            train_noisy_loaders.append(train_noisy_loader)

            valid_noisy_loader = DataLoader(self.valid_noisy_dataset, 
                                    batch_size=self.config['valid_batch_size'], 
                                    num_workers=self.config['num_workers'],
                                    shuffle=False,
                                    sampler=SubsetRandomSampler(valid_task_data_list[i]))
            valid_noisy_loaders.append(valid_noisy_loader)
        self.train_loaders = train_noisy_loaders
        self.valid_loaders = valid_noisy_loaders
        self.create_chosen_dataloaders()

    def create_semi(self):
        self.train_semi_dataset, self.valid_semi_dataset, self.semi_dataset = extract_semi_data(self.train_dataset, 
                                                self.valid_dataset, 
                                                anomaly_num=self.config['semi_anomaly_num'], 
                                                anomaly_overlap=self.config['semi_overlap'])                                                    
        train_task_data_list = self.train_semi_dataset.sample_indices_in_task
        valid_task_data_list = self.valid_semi_dataset.sample_indices_in_task 
        train_semi_loaders, valid_semi_loaders = [], []
        for i in range(self.config['num_task']):
            train_semi_loader = DataLoader(self.train_semi_dataset,
                                    batch_size=self.config['train_batch_size'],
                                    num_workers=self.config['num_workers'],
                                    sampler=SubsetRandomSampler(train_task_data_list[i]))
            train_semi_loaders.append(train_semi_loader)

            valid_semi_loader = DataLoader(self.valid_semi_dataset, 
                                    batch_size=self.config['valid_batch_size'], 
                                    num_workers=self.config['num_workers'],
                                    shuffle=False,
                                    sampler=SubsetRandomSampler(valid_task_data_list[i]))
            valid_semi_loaders.append(valid_semi_loader)
        self.train_loaders = train_semi_loaders
        self.valid_loaders = valid_semi_loaders
        self.create_chosen_dataloaders()

    def create_continual(self): 
        self.create_chosen_dataloaders()

    def create_transfer(self): 
        self.train_transfer_dataset = extract_fewshot_data(self.train_dataset, self.config['transfer_target_sample_num'])
        train_transfer_task_data_list = self.train_transfer_dataset.sample_indices_in_task
        self.train_transfer_loaders = []
        for i in range(self.config['num_task']):
            train_transfer_loader = DataLoader(self.train_transfer_dataset,
                                    batch_size=self.config['train_batch_size'],
                                    num_workers=self.config['num_workers'],
                                    sampler=SubsetRandomSampler(train_transfer_task_data_list[i]),
                                    drop_last=False)
            self.train_transfer_loaders.append(train_transfer_loader)
        self.create_chosen_transfer_dataloaders()
        
    def create_chosen_dataloaders(self):
        if self.config['model'] == 'dra':
            for idx in self.config['train_task_id']:
                self.chosen_train_loaders.append([self.train_loaders[idx], self.refer_loaders[idx]])
            for idx in self.config['valid_task_id']:
                self.chosen_valid_loaders.append([self.valid_loaders[idx], self.refer_loaders[idx]])
                self.chosen_vis_loaders.append([self.vis_loaders[idx], self.refer_loaders[idx]])
        else:
            for idx in self.config['train_task_id']:
                self.chosen_train_loaders.append(self.train_loaders[idx])
            for idx in self.config['valid_task_id']:
                self.chosen_valid_loaders.append(self.valid_loaders[idx])
                self.chosen_vis_loaders.append(self.vis_loaders[idx])

    def create_chosen_transfer_dataloaders(self): 
        if self.config['model'] == 'dra':
            for idx in self.config['train_task_id']: # for step 1, train source task 
                self.chosen_train_loaders.append([self.train_loaders[idx], self.refer_loaders[idx]])
                self.chosen_valid_loaders.append([self.valid_loaders[idx], self.refer_loaders[idx]])
                self.chosen_vis_loaders.append([self.vis_loaders[idx], self.refer_loaders[idx]])
            for idx in self.config['valid_task_id']: # for step 2, train target task
                self.chosen_transfer_train_loaders.append([self.train_transfer_loaders[idx], self.refer_loaders[idx]])
                self.chosen_transfer_valid_loaders.append([self.valid_loaders[idx], self.refer_loaders[idx]])
                self.chosen_transfer_vis_loaders.append([self.vis_loaders[idx], self.refer_loaders[idx]])
        else:
            for idx in self.config['train_task_id']: # for step 1, train source task
                self.chosen_train_loaders.append(self.train_loaders[idx])
                self.chosen_valid_loaders.append(self.valid_loaders[idx])
                self.chosen_vis_loaders.append(self.vis_loaders[idx])
            for idx in self.config['valid_task_id']: # for step 2, train target task
                self.chosen_transfer_train_loaders.append(self.train_transfer_loaders[idx])
                self.chosen_transfer_valid_loaders.append(self.valid_loaders[idx])
                self.chosen_transfer_vis_loaders.append(self.vis_loaders[idx])