import torch
import torch.nn.functional as F
from collections import OrderedDict
from scipy.ndimage import gaussian_filter
from arch.base import ModelBase
from torchvision import models

__all__ = ['SPADE']

class SPADE(ModelBase):
    def __init__(self, config):
        super(SPADE, self).__init__(config)
        self.config = config

        if self.config['net'] == 'resnet18': 
            self.net = models.resnet18(pretrained=True, progress=True).to(self.device) 
            
        self.features = []
        self.get_layer_features()
        
    def get_layer_features(self):
        
        def hook_t(module, input, output):
            self.features.append(output)
        
        self.net.layer1[-1].register_forward_hook(hook_t)
        self.net.layer2[-1].register_forward_hook(hook_t)
        self.net.layer3[-1].register_forward_hook(hook_t)
        self.net.avgpool.register_forward_hook(hook_t)
    
    @staticmethod
    def cal_distance_matrix(x, y):
        """Calculate Euclidean distance matrix with torch.tensor"""
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))
        return dist_matrix

    def train_model(self, train_loader, task_id, inf=''):
        self.net.eval()
        self.train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])

        for _ in range(self.config['num_epochs']):
            for batch_id, batch in enumerate(train_loader):
                img = batch['img'].to(self.device) 

                self.features.clear()
                with torch.no_grad():
                    _ = self.net(img)
                    
                # get the intermediate layer outputs
                for k,v in zip(self.train_outputs.keys(), self.features):
                    self.train_outputs[k].append(v)
                # initialize hook outputs
                self.features = [] 

            for k, v in self.train_outputs.items():
                self.train_outputs[k] = torch.cat(v, 0)
                
    def prediction(self, valid_loader, task_id):
        self.net.eval()
        self.clear_all_list()
        self.test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])

        for batch_id, batch in enumerate(valid_loader):
            img = batch['img'].to(self.device)
            mask = batch['mask'].to(self.device)
            label = batch['label'].to(self.device)
            mask[mask>=0.5] = 1
            mask[mask<0.5] = 0
            self.img_gt_list.append(label.cpu().detach().numpy()[0])
            self.pixel_gt_list.append(mask.cpu().detach().numpy()[0,0])
            self.img_path_list.append(batch['img_src'])

            # extract features from backbone
            with torch.no_grad():
                self.features.clear()
                _ = self.net(img)

            for k,v in zip(self.test_outputs.keys(), self.features):
                self.test_outputs[k].append(v)
            # initialize hook outputs
            self.features = [] 
        
        for k, v in self.test_outputs.items():
            self.test_outputs[k] = torch.cat(v, 0)
        
        # load train feature 
        # self.train_outputs = load_feat_pickle(feat=self.train_outputs, file_path=self.embedding_dir_path)

        # calculate distance matrix
        dist_matrix = SPADE.cal_distance_matrix(torch.flatten(self.test_outputs['avgpool'], 1), 
                                                torch.flatten(self.train_outputs['avgpool'], 1))
        
        # select K nearest neighbor and take advantage 
        topk_values, topk_indexes = torch.topk(dist_matrix, k=self.config['_top_k'], dim=1, largest=False)
        scores = torch.mean(topk_values, 1).cpu().detach().numpy()
        self.img_pred_list = scores

        # score_map_list = []
        for t_idx in range(self.test_outputs['avgpool'].shape[0]):
            score_maps = []
            # for each layer
            for layer_name in ['layer1', 'layer2', 'layer3']:
                # construct a gallery of features at all pixel locations of the K nearest neighbors
                topk_feat_map = self.train_outputs[layer_name][topk_indexes[t_idx]]
                test_feat_map = self.test_outputs[layer_name][t_idx:t_idx + 1]
                feat_gallery = topk_feat_map.transpose(3, 1).flatten(0, 2).unsqueeze(-1).unsqueeze(-1)

                # calculate distance matrix
                dist_matrix_list = []

                for d_idx in range(feat_gallery.shape[0] // 100):
                    dist_matrix = torch.pairwise_distance(feat_gallery[d_idx * 100:d_idx * 100 + 100], test_feat_map)
                    dist_matrix_list.append(dist_matrix)
                dist_matrix = torch.cat(dist_matrix_list, 0)

                # k nearest features from the gallery (k=1)
                score_map = torch.min(dist_matrix, dim=0)[0]
                score_map = F.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=self.config['data_crop_size'],
                                          mode='bilinear', align_corners=False)
                score_maps.append(score_map)
            
            # average distance between the features
            score_map = torch.mean(torch.cat(score_maps, 0), dim=0)

            # apply gaussian smoothing on the score map
            score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=4)
            self.pixel_pred_list.append(score_map)

        


        

