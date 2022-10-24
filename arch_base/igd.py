import torch
import torch.nn as nn
from models.igd.ssim_module import *
from models.igd.mvtec_module import *
from pytorch_msssim import ms_ssim, ssim
from tools.utils import create_folders
from sklearn.metrics import roc_curve, auc

__all__ = ['IGD']

class IGD():
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        self.config = config
        self.device = device
        self.file_path = file_path
        self.scheduler = scheduler
        self.generator = self.net['g'].to(self.device)
        self.discriminator = self.net['d'].to(self.device)
        self.optimizer_g = optimizer[0]
        self.optimizer_d = optimizer[1]
        self.mse_criterion = torch.nn.MSELoss()
        self.l1_criterion = torch.nn.L1Loss()
        self.bce_criterion = torch.nn.BCELoss()
        self.sigbce_criterion = torch.nn.BCEWithLogitsLoss()
        self.img_gt_list = []
        self.img_pred_list = []

    def init_c(self, data_loader, generator, eps=0.1):
        generator.c = None
        c = torch.zeros(1, self.config['latent_dimension']).to(self.device)
        generator.eval()
        n_samples = 0
        with torch.no_grad():
            for index, (images, label) in enumerate(data_loader):
                # get the inputs of the batch
                img = images.to(self.device)
                outputs = generator.encoder(img)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        
        c /= n_samples 

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def init_sigma(self, data_loader, generator, sig_f=1):
        generator.sigma = None
        generator.eval()
        tmp_sigma = torch.tensor(0.0, dtype=torch.float).to(self.device)
        n_samples = 0
        with torch.no_grad():
            for index, (images, label) in enumerate(data_loader):
                img = images.to(self.device)
                latent_z = self.generator.encoder(img)
                diff = (latent_z - self.generator.c) ** 2
                tmp = torch.sum(diff.detach(), dim=1)
                if (tmp.mean().detach() / sig_f) < 1:
                    tmp_sigma += 1
                else:
                    tmp_sigma += tmp.mean().detach() / sig_f
                n_samples += 1
        tmp_sigma /= n_samples
        return tmp_sigma
    
    def train_model(self, train_loaders, inf=''):
        ck_path = 'checkpoints'
        create_folders(ck_path)
        ck_path = ck_path + 'igd' 
        create_folders(ck_path)
        
        for task_idx, train_loader in enumerate(train_loaders):

            print('run task: {}'.format(self.config['train_task_id'][task_idx]))
            task_ck_path = ck_path + str(task_idx)
            create_folders(task_ck_path)
            #AUC_LIST = [] 
            #global test_auc
            #BEST_AUC = 0
            #test_auc = 0

            self.generator.c = None
            self.generator.sigma = None


            self.generator.train()
            self.discriminator.train()

            for param in self.generator.pretrain.parameters():
                param.requires_grad = False
        
            START_ITER = 0
            END_ITER = int(train_size / self.config['train_batch_size'] * self.config['max_epoch']) 
            for epoch in range(self.config['num_epochs']):
                for batch_id, batch in enumerate(train_loader):
                    train_size = len(train_loader)
                    iteration = int(train_size / self.config['train_batch_size'] * epoch)

                    self.generator.c = self.init_c(train_loader, self.generator)
                    self.generaotr.c.requries_grad = False
                    self.generator.sigma = self.init_sigma(train_loader, self.generator)
                    self.generator.requires_grad = False

                    poly_lr_scheduler(self.optimizer_d, init_lr=self.config['lr'], iter=iteration, max_iter=END_ITER)
                    poly_lr_scheduler(self.optimizer_g, init_lr=self.config['lr'], iter=iteration, max_iter=END_ITER)
                    
                    real_data = batch['img'].to(self.device)
                    b, c, _, _ = real_data.shape()

                    # Train E
                    self.optimizer_g.zero_grad()
                    latent_z = self.generator.encoder(real_data)
                    fake_data = self.generator.generate(latent_z)

                    # Reconstruction Loss
                    weight = 0.85
                    ms_ssim_batch_wise = 1 - ms_ssim(real_data, fake_data, data_range=self.config['data_range'],
                                                     size_average=True, win_size=11, 
                                                     weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
                    
                    l1_batch_wise = self.l1_criterion(real_data, fake_data) / self.config['data_range']
                    ms_ssim_l1 = weight * ms_ssim_batch_wise + (1 - weight) * l1_batch_wise

                    ####### Interpolate ###########
                    e1 = torch.flip(latent_z, dims=[0])
                    alpha = torch.FloatTensor(b, 1).uniform_(0, 0.5).to(self.device)
                    e2 = alpha * latent_z + (1 - alpha) * e1
                    g2 = self.generator.generate(e2)
                    reg_inter = torch.mean(self.discriminator(g2) ** 2)

                    ############ GAC ############
                    diff = (latent_z - self.generator.c) ** 2
                    dist = -1 * (torch.sum(diff, dim=1) / self.generator.sigma)
                    svdd_loss = torch.mean(1 - torch.exp(dist))

                    encoder_loss = ms_ssim_l1 + svdd_loss + 0.1 * reg_inter
                    encoder_loss.backward()
                    self.optimizer_g.step()

                    ############ Discriminator ############
                    self.optimizer_d.zero_grad()
                    g2 = self.generator.generate(e2).detach()
                    fake_data = self.generator(real_data).detach()
                    d_loss_front = torch.mean((self.discriminator(g2) - alpha) ** 2)
                    gamma = 0.2
                    tmp = fake_data + gamma * (real_data - fake_data)
                    d_loss_back = torch.mean(self.discriminator(tmp) ** 2)
                    d_loss = d_loss_front + d_loss_back
                    d_loss.backward()
                    self.optimizer_d.step()

                    if iteration % int((train_size / self.config['train_batch_size']) * 10) == 0 and iteration != 0:
                        self.generator.sigma = self.init_sigma(self.train_loader, self.generator)
                        self.generator.c = self.init_c(self.train_loader, self.generator)
                    
                    if iteration % int((train_size / self.config['train_batch_size']) * 5) == 0 and iteration == END_ITER:
                        
                        g_ck = task_ck_path + 'g' 
                        create_folders(g_ck)
                        d_ck  = task_ck_path + 'd'
                        create_folders(d_ck)

                        torch.save(self.generator.state_dict(), g_ck)
                        torch.save(self.discriminator.state_dict(), d_ck)
                        #torch.save(self.optimizer_g.state_dict())
                        #torch.save(self.optimizer_d.state_dict())

    def prediction(self, valid_loader):
        self.generator.eval()
        self.discriminator.eval()
        self.img_gt_list.clear()
        self.img_pred_list.clear()
         
        normal_gsvdd = []
        abnormal_gsvdd = []
        normal_recon = []
        abnormal_recon = []

        with torch.no_grad():
            for batch_id, batch in enumerate(valid_loader):
                img = batch['img'].to(self.device)
                mask = batch['mask'].to(self.device)
                label = batch['label'].to(self.device)

                latent_z = self.generator.encoder(img)
                generate_result = self.generator(img)

                ################ Normal ##############

                for visual_index in range(latent_z.shape[0]):
                    weight = 0.85
                    tmp_org_unsq = img[visual_index].unsqueeze(0)
                    tmp_rec_unsq = generate_result[visual_index].unsqueeze(0)
                    ms_ssim_batch_wise = 1 - ms_ssim(tmp_org_unsq, tmp_rec_unsq, 
                                                    data_range=self.config['data_range'],
                                                    size_average=True, win_size=11, 
                                                    weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
                    
                    l1_loss = self.l1_criterion(img[visual_index], generate_result[visual_index]) / self.config['data_range']
                    ms_ssim_l1 = weight * ms_ssim_batch_wise + (1 - weight) * l1_loss

                    diff = (latent_z[visual_index] - self.generator.c) ** 2
                    dist = -1 * torch.sum(diff, dim=1) / self.generator.sigma
                    guass_svdd_loss = 1 - torch.exp(dist)
                    anomaly_score = (0.5 * ms_ssim_l1 + 0.5 * guass_svdd_loss).cpu().detach().numpy()

                    self.img_pred_list.append(anomaly_score)
                    
                    la = label[visual_index]

                    if la == 'good':
                        self.img_gt_list.append(0)
                    else:
                        self.img_gt_list.append(1)


                    
                    