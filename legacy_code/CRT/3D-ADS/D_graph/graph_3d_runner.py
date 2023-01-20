from data.mvtec3d import get_data_loader
import torch
from torch import nn
from tqdm import tqdm
from D_graph.gnn_model import CONAD_Base
from torch_geometric.data import Data
import numpy as np
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score 
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import D_graph.gnn_model.util as util
import torch.nn.functional as F
from torch_scatter import scatter
from copy import deepcopy
import cv2 as cv
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
from D_graph.model import GCNNet,GCNNet1
from numpy import random
from D_graph.au_pro import calculate_au_pro

def visualize_graph(G,color):
    plt.figure(figsize=(7,7))
    myfig = plt.gcf()
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G,pos=nx.spring_layout(G,seed=42),with_labels=False,node_color=color,cmap="Set2")
    myfig.savefig('/ssd2/m3lab/usrs/crt/3D-ADS/figure.png',dpi=300)
    # plt.show()


class GNN_runner():
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.downsample_ratio = 7
        self.avg_pooling = nn.AvgPool2d(kernel_size=self.downsample_ratio)
        self.k_neig = 6
        self.lr = 1e-3
        self.weight_decay = 5e-4

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_features = int(self.downsample_ratio*self.downsample_ratio*3)

        self.first_train = True
        if self.first_train:
            self.model = GCNNet(self.input_features).to(self.device)
            # self.model = torch.load('/ssd2/m3lab/usrs/crt/3D-ADS/D_graph/gnn_pt/gnn_net_10_all.pth')

            self.first_train = False

        self.img_pred_list = [] # list<numpy>
        self.img_gt_list = [] # list<numpy>
        self.pixel_pred_list = [] # list<numpy(m,n)>
        self.pixel_gt_list = [] # list<numpy(m,n)>
        self.pixel_pred_label_list = []
        
    def fit(self, class_name):
        
        train_loader = get_data_loader("train", class_name=class_name, img_size=self.image_size)

        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        k = 0
        total_loss = 0
        total_correct = []
        
        for sample, _ in tqdm(train_loader, desc=f'Extracting train features for class {class_name}'):
            Data_xyz = sample[1]
            x = []
            for i in range(self.downsample_ratio):
                for j in range(self.downsample_ratio):
                    x1 = Data_xyz[:,:,i::self.downsample_ratio,j::self.downsample_ratio]
                    x.append(x1)
            Data_xyz = torch.cat(x,dim=1)
            Data_xyz = Data_xyz.permute(0,2,3,1)
            Data_xyz = Data_xyz.view(1,-1,3*self.downsample_ratio*self.downsample_ratio)
            # print("data_xyz:",Data_xyz.shape)

            for x in Data_xyz:
                # print("x:",x)
                self.model.train()
                edge_index = util.construct_H_with_KNN(x, k_neig = self.k_neig)
                train_mask = [1 if random.random()>0.2 else 0 for i in range(int(self.image_size*self.image_size/(self.downsample_ratio*self.downsample_ratio)))]
                test_mask = [1 if i==0 else 0 for i in train_mask]
                # print(len(train_mask))
                # print(len(test_mask))
                # y = torch.zeros((self.image_size//self.downsample_ratio*self.image_size//self.downsample_ratio,1), dtype=torch.float)
                # y = y.bool()
                # print(y)
                y_data = [1 if random.random()>0.5 else 0 for i in range(x.shape[0])]
                y_data = torch.tensor(y_data)
                mask = y_data.view(-1,1)
                mask = mask>0
                mask = mask.repeat(1, x.shape[1])
                x_negetive = x.mul(0.2+torch.rand(x.shape[0],x.shape[1])*0.4)
                x_data = x_negetive.masked_scatter_(mask,x)
                # print(x_data)
                # print(y_data)
                graph = Data(x=x_data, edge_index=edge_index,y=y_data,train_mask=train_mask,test_mask=test_mask).to(self.device)
                # print("graph:",graph)
                optimizer.zero_grad()
                out = self.model(graph)
                # print(out)
                loss = F.nll_loss(out[graph.train_mask], graph.y[graph.train_mask])
                total_loss = total_loss + loss
                loss.backward()
                optimizer.step()
                # evaluation
                self.model.eval()
                pred = self.model(graph)
                pred = pred.argmax(dim=1)
                # print("pred_argmax:",pred)
                correct = (pred[graph.test_mask] == graph.y[graph.test_mask]).sum()
                # print(correct)
                correct = correct/pred.shape[0]
                # print(correct.cpu().numpy())
                total_correct.append(correct.cpu().numpy())



            if k%65 == 64:
                print("total loss",total_loss/65)
                print("total correct",np.mean(np.array(total_correct)))
                total_loss = 0
                total_correct = []
            k = k+1
        torch.save(self.model,'/ssd2/m3lab/usrs/crt/3D-ADS/D_graph/gnn_pt/gnn_net_GCN_100_cookie.pth')
        return self

    def test_fit(self, class_name):
        m = nn.MaxPool2d((self.downsample_ratio,self.downsample_ratio), stride=self.downsample_ratio)
        test_loader = get_data_loader("test", class_name=class_name, img_size=self.image_size)

        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        k = 0
        total_loss = 0
        
        for sample, mask, label in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
            # print("mask:",mask)
            # print("mask_shape:",mask.shape)
            mask = m(mask)
            mask = mask.reshape(self.image_size//self.downsample_ratio*self.image_size//self.downsample_ratio)
            # print("mask:",mask.shape)
            # print("mask:",mask)

            Data_xyz = sample[1]
            x = []
            for i in range(self.downsample_ratio):
                for j in range(self.downsample_ratio):
                    x1 = Data_xyz[:,:,i::self.downsample_ratio,j::self.downsample_ratio]
                    x.append(x1)
            Data_xyz = torch.cat(x,dim=1)
            Data_xyz = Data_xyz.permute(0,2,3,1)
            Data_xyz = Data_xyz.view(1,-1,3*self.downsample_ratio*self.downsample_ratio)
            

            for x in Data_xyz:
                edge_index = util.construct_H_with_KNN(x, k_neig = self.k_neig)
                # y = torch.zeros((self.image_size//self.downsample_ratio*self.image_size//self.downsample_ratio,1), dtype=torch.float)
                # y = y.bool()
                # # print(y)
                y_data = mask.long()
                # print("y_data:",y_data)
                graph = Data(x=x, edge_index=edge_index,y=y_data).to(self.device)
                # print("graph:",graph)
                optimizer.zero_grad()
                out = self.model(graph)
                # print(out)
                loss = F.nll_loss(out, graph.y)
                total_loss = total_loss + loss
                loss.backward()
                optimizer.step()
            if k%80 == 79:
                print(total_loss/80)
                total_loss = 0
            k = k+1
        # torch.save(self.model,'/ssd2/m3lab/usrs/crt/3D-ADS/D_graph/gnn_pt/gnn_net1_test_15.pth')
        return self

    def eval(self,class_name):
        train_loader = get_data_loader("train", class_name=class_name, img_size=self.image_size)

        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        k = 0
        total_correct = []

        self.model.eval()
        with torch.no_grad():
            for sample, _ in tqdm(train_loader, desc=f'Extracting test features for class {class_name}'):
                Data_xyz = sample[1]
                mask = mask.reshape(224,224)
                mask = mask.cpu().numpy()

                x = []
                for i in range(self.downsample_ratio):
                    for j in range(self.downsample_ratio):
                        x1 = Data_xyz[:,:,i::self.downsample_ratio,j::self.downsample_ratio]
                        x.append(x1)
                Data_xyz = torch.cat(x,dim=1)
                Data_xyz = Data_xyz.permute(0,2,3,1)
                Data_xyz = Data_xyz.view(1,-1,3*self.downsample_ratio*self.downsample_ratio)

                for x in Data_xyz:
                    edge_index = util.construct_H_with_KNN(x, k_neig = self.k_neig)
                    y = torch.zeros((self.image_size//self.downsample_ratio*self.image_size//self.downsample_ratio,1), dtype=torch.float)
                    y = y.bool()
                    # print(y)
                    y_data = [1 if i else 0 for i in y]
                    # print(len(y_data))
                    y_data = torch.tensor(y_data)
                    graph = Data(x=x, edge_index=edge_index,y=y_data).to(self.device)
                    pred = self.model(graph)
                    print("pred:",pred)
                    print("pred_np:",np.exp(pred.cpu().numpy()))

                    print("pred_max:",pred.max(dim=1))
                    pred = pred.argmax(dim=1)
                    print("pred_argmax:",pred)
                    correct = (pred == graph.y).sum()
                    # print(correct)
                    correct = correct/pred.shape[0]
                    # print(correct.cpu().numpy())
                    total_correct.append(correct.cpu().numpy())
                    
                # if k%10 == 0:
                #     print(total_correct/10)
                #     total_correct = 0
                # k = k+1
            print(np.mean(total_correct))
        return self

    def evaluate(self, class_name):
        # raise NotImplementedError
        # image_rocaucs = dict()
        # pixel_rocaucs = dict()
        # au_pros = dict()
        self.model = torch.load('/ssd2/m3lab/usrs/crt/3D-ADS/D_graph/gnn_pt/gnn_net_GCN_100_cookie.pth')
        test_loader = get_data_loader("test", class_name=class_name, img_size=self.image_size)
        with torch.no_grad():
            total_auc = 0
            total_ap = 0
            y_true = []
            img_y_true = []
            y_pre = []
            img_y_pre = []
            
            for sample, mask, label in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
                Data_xyz = sample[1]
                # mask = self.avg_pooling(mask)
                # mask = mask.view(-1,1)
                # mask = torch.where(mask > 0.5, 1., .0)
                # print("mask:",mask)
                # print("mask_shape:",mask.shape)
                mask = mask.reshape(224,224)
                # print("mask:",mask.shape)
                mask[mask >= 0.5] = 1
                mask[mask < 0.5] = 0
                mask = mask.cpu().numpy()
                # print("mask:",mask)

                x = []
                for i in range(self.downsample_ratio):
                    for j in range(self.downsample_ratio):
                        x1 = Data_xyz[:,:,i::self.downsample_ratio,j::self.downsample_ratio]
                        x.append(x1)

                Data_xyz = torch.cat(x,dim=1)
                Data_xyz = Data_xyz.permute(0,2,3,1)
                Data_xyz = Data_xyz.view(1,-1,3*self.downsample_ratio*self.downsample_ratio)
                for x in Data_xyz:
                    edge_index = util.construct_H_with_KNN(x, k_neig = self.k_neig)
                    # edge_index = self._get_edge_index(self.image_size//self.downsample_ratio)
                    # y = mask.bool()
                    graph = Data(x=x, edge_index=edge_index).to(self.device)
                    
                    pred = self.model(graph)
                    pred = np.exp(pred.cpu().numpy())
                    pred_normal = pred[:,0]
                    pred = pred[:,1]
                    
                    # print("no_process_pre_label:",pred.max())
                    # print("pred:",pred)
                    pred = pred.reshape(int(self.image_size/self.downsample_ratio),int(self.image_size/self.downsample_ratio))
                    # print("pred_reshape:",pred.shape)
                    y = cv.resize(pred,(224,224))
                    # y = gaussian_filter(map_resized, sigma=4)
                    # y_label = np.where(y > 0.5, 1., .0)
                    # print("label:",label.cpu().numpy()[0])
                    # print("pre_anormal_label_min:",y.min())
                    # print("pre_anormal_label_max:",y.max())
                    # print("pre_narmal_label_min:",pred_normal.min())
                    # print("pre_narmal_label_max:",pred_normal.max())
                    # print("label:",label.cpu().numpy()[0])

                    self.pixel_gt_list.extend(mask.flatten().astype(int))
                    self.pixel_pred_list.extend(y.flatten())

                    self.img_gt_list.append(label.cpu().numpy()[0])
                    self.img_pred_list.append(y.max())

                    # print("pixel_auroc:",roc_auc_score(mask.flatten(),y.flatten()))
                    # print("pixel_ap:",average_precision_score(mask.flatten(),y_label.flatten()))
                    # print("img_auroc:",roc_auc_score(label.cpu().numpy(),[y.max()]))
                    # print("img_ap:",average_precision_score(label.cpu().numpy(),[1. if y.max()>0.5 else .0]))
            # print(self.img_gt_list)
            # print(self.img_pred_list)
            pixel_auroc = roc_auc_score(self.pixel_gt_list, self.pixel_pred_list)
            img_auroc = roc_auc_score(self.img_gt_list, self.img_pred_list)

            pixel_ap = average_precision_score(self.pixel_gt_list, self.pixel_pred_list)
            img_ap = average_precision_score(self.img_gt_list, self.img_pred_list)

            pixel_pro = 1
            # pixel_pro, pro_curve = calculate_au_pro(self.pixel_gt_list, self.pixel_pred_list)
                
        return pixel_auroc, img_auroc, pixel_ap, img_ap,pixel_pro
