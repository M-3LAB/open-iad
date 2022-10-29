import torch
from arch_base.base import ModelBase
from models.fastflow.func import AverageMeter
from ignite.contrib import metrics

__all__ = ['FastFlow']

class FastFlow(ModelBase):
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
        self.net.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_model(self, train_loader, inf=''):
        self.net.train()
        loss_meter = AverageMeter() 
        for epoch in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                img = batch['img'].to(self.device)
                ret = self.net(img)
                loss = ret['loss']
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_meter.update(loss.item())

            infor = '\r{}[Epoch {}/{}] Loss: {:.6f}'.format(inf, epoch, self.config['num_epochs'],
                                                            loss_meter.avg) 
            print(infor, flush=True, end='')
    
    def prediction(self, valid_loader, task_id=None):
        self.net.eval()
        auroc_metric = metrics.ROC_AUC()
        with torch.no_grad():
            for batch_id, batch in enumerate(valid_loader):
                img = batch['img'].to(self.device)
                mask = batch['mask']
                ret = self.net(img)
                mask = mask.flatten()

                outputs = ret["anamaly_map"].cpu().detach()
                outputs = outputs.flatten()
                auroc_metric.update((outputs, mask))
        
        pixel_auroc = auroc_metric.compute()
        img_auroc = None
        return pixel_auroc, img_auroc 
        
    