import torch
import argparse
from arch.base import ModelBase
from models.igd.net_igd import NetIGD
from models.igd.ssim_module import *
from models.igd.mvtec_module import *
from pytorch_msssim import ms_ssim
from optimizer.optimizer import get_optimizer

__all__ = ['IGD']

class IGD(ModelBase):
    def __init__(self, config):
        super(IGD, self).__init__(config)
        self.config = config

        args = argparse.Namespace(**self.config) 
        self.net = NetIGD(args)
        self.optimizer_g = get_optimizer(self.config, self.net.g.parameters())
        self.optimizer_d = get_optimizer(self.config, self.net.d.parameters())
        self.scheduler_g = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_g, [args.num_epochs * 0.8, args.num_epochs * 0.9], gamma=args._gamma, last_epoch=-1)
        self.scheduler_d = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_d, [args.num_epochs * 0.8, args.num_epochs * 0.9], gamma=args._gamma, last_epoch=-1)
        self.optimizer = [self.optimizer_g, self.optimizer_d]
        self.scheduler = [self.scheduler_g, self.scheduler_d] 
        
        self.generator = self.net.g.to(self.device)
        self.discriminator = self.net.d.to(self.device)

        self.mse_criterion = torch.nn.MSELoss()
        self.l1_criterion = torch.nn.L1Loss()
        self.bce_criterion = torch.nn.BCELoss()
        self.sigbce_criterion = torch.nn.BCEWithLogitsLoss()

    def init_c(self, data_loader, generator, eps=0.1):
        generator.c = None
        c = torch.zeros(1, self.config['_latent_dimension']).to(self.device)
        generator.eval()
        n_samples = 0
        with torch.no_grad():
            for index, batch in enumerate(data_loader):
                img = batch['img'].to(self.device)

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
            for index, batch in enumerate(data_loader):
                img = batch['img'].to(self.device)
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
    
    def train_model(self, train_loader, task_id, inf=''):
        self.generator.c = None
        self.generator.sigma = None

        self.generator.train()
        self.discriminator.train()

        for param in self.generator.pretrain.parameters():
            param.requires_grad = False
        
        START_ITER = 0
        train_size = len(train_loader)
        END_ITER = int(train_size / self.config['train_batch_size'] * self.config['_max_epoch']) 
        for epoch in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                train_size = len(train_loader)
                iteration = int(train_size / self.config['train_batch_size'] * epoch)

                self.generator.c = self.init_c(train_loader, self.generator)
                self.generator.c.requries_grad = False
                self.generator.sigma = self.init_sigma(train_loader, self.generator)
                self.generator.requires_grad = False

                poly_lr_scheduler(self.optimizer_d, init_lr=self.config['_base_lr'], iter=iteration, max_iter=END_ITER)
                poly_lr_scheduler(self.optimizer_g, init_lr=self.config['_base_lr'], iter=iteration, max_iter=END_ITER)
                    
                real_data = batch['img'].to(self.device)
                b, c, _, _ = real_data.size()

                # Train E
                self.optimizer_g.zero_grad()
                latent_z = self.generator.encoder(real_data)
                fake_data = self.generator.generate(latent_z)

                # Reconstruction Loss
                weight = 0.85
                ms_ssim_batch_wise = 1 - ms_ssim(real_data, fake_data, data_range=self.config['_data_range'],
                                                    size_average=True, win_size=11, 
                                                    weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
                    
                l1_batch_wise = self.l1_criterion(real_data, fake_data) / self.config['_data_range']
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


    def prediction(self, valid_loader, task_id):
        self.generator.eval()
        self.discriminator.eval()
        self.clear_all_list()

        with torch.no_grad():
            for batch_id, batch in enumerate(valid_loader):
                img = batch['img'].to(self.device)
                label = batch['label']

                latent_z = self.generator.encoder(img)
                generate_result = self.generator(img)

                ################ Normal ##############
                for visual_index in range(latent_z.shape[0]):
                    weight = 0.85
                    tmp_org_unsq = img[visual_index].unsqueeze(0)
                    tmp_rec_unsq = generate_result[visual_index].unsqueeze(0)
                    ms_ssim_batch_wise = 1 - ms_ssim(tmp_org_unsq, tmp_rec_unsq, 
                                                     data_range=self.config['_data_range'],
                                                     size_average=True, win_size=11, 
                                                     weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
                    
                    l1_loss = self.l1_criterion(img[visual_index], generate_result[visual_index]) / self.config['_data_range']
                    ms_ssim_l1 = weight * ms_ssim_batch_wise + (1 - weight) * l1_loss

                    diff = (latent_z[visual_index] - self.generator.c) ** 2
                    dist = -1 * torch.sum(diff, dim=1) / self.generator.sigma
                    guass_svdd_loss = 1 - torch.exp(dist)
                    anomaly_score = (0.5 * ms_ssim_l1 + 0.5 * guass_svdd_loss).cpu().detach().numpy()
                    self.img_pred_list.append(float(anomaly_score))
                    self.img_path_list.append(batch['img_src'])

                self.img_gt_list.extend(label.numpy().tolist())


                    
                    