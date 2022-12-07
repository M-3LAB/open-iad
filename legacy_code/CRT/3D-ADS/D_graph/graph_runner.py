from data.mvtec3d import get_data_loader
import torch
from torch import nn
from tqdm import tqdm
from pygod.models import CONAD
from pygod.models import DOMINANT 
from torch_geometric.data import Data
import numpy as np
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score 



class GNN_runner():
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.methods = CONAD(epoch=2,verbose=False)
        self.downsample_ratio = 7
        self.avg_pooling = nn.AvgPool2d(kernel_size=self.downsample_ratio)
        # self.methods = DOMINANT(epoch=5,verbose=True)
        
    def fit(self, class_name):
        train_loader = get_data_loader("train", class_name=class_name, img_size=self.image_size)
        k = 0
        for sample, _ in tqdm(train_loader, desc=f'Extracting train features for class {class_name}'):
            Data_xyz = sample[1]
            x = []
            for i in range(self.downsample_ratio):
                for j in range(self.downsample_ratio):
                    x1 = Data_xyz[:,:,i::self.downsample_ratio,j::self.downsample_ratio]
                    # x1 = x1.unsqueeze(dim=1)
                    x.append(x1)

            Data_xyz = torch.cat(x,dim=1)
            # print('torch.cat:',Data_xyz.shape)
            Data_xyz = Data_xyz.permute(0,2,3,1)
            # print('torch.permute:',Data_xyz.shape)
            Data_xyz = Data_xyz.view(1,-1,3*self.downsample_ratio*self.downsample_ratio)
            # print('torch.view:',Data_xyz.shape)

            for x in Data_xyz:
                # print(x)
                edge_index = self._get_edge_index(self.image_size//self.downsample_ratio)
                y = torch.zeros((self.image_size//self.downsample_ratio*self.image_size//self.downsample_ratio,1), dtype=torch.float)
                y = y.bool()
                graph = Data(x=x, edge_index=edge_index,y=y)
                # print(graph)
            self.methods = self._train_anomaly_detector(self.methods,graph)
            # print(f"Epoch: {k} Loss: {self.methods.loss}")
            # k = k+1
        print(f"class: {class_name} Loss: {self.methods.loss}")
        torch.save(self.methods.model, '/ssd2/m3lab/usrs/crt/3D-ADS/model_saved/tire_gnn_10_1_7.pt')
        
        return self

    def evaluate(self, class_name):
        # raise NotImplementedError
        # image_rocaucs = dict()
        # pixel_rocaucs = dict()
        # au_pros = dict()
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
                mask = self.avg_pooling(mask)
                mask = mask.view(-1,1)
                mask = torch.where(mask > 0.5, 1., .0)
                # print(torch.where(mask==1))
                # print(mask)
                # print(mask.shape)
                # print('type(mask): ',type(mask))
                # print('mask.dtype: ',mask.dtype)
                x = []
                for i in range(self.downsample_ratio):
                    for j in range(self.downsample_ratio):
                        x1 = Data_xyz[:,:,i::self.downsample_ratio,j::self.downsample_ratio]
                        # x1 = x1.unsqueeze(dim=1)
                        x.append(x1)

                Data_xyz = torch.cat(x,dim=1)
                # print('torch.cat:',Data_xyz.shape)
                Data_xyz = Data_xyz.permute(0,2,3,1)
                # print('torch.permute:',Data_xyz.shape)
                Data_xyz = Data_xyz.view(1,-1,3*self.downsample_ratio*self.downsample_ratio)
                # print('torch.view:',Data_xyz.shape)

                for x in Data_xyz:
                    # print(x)
                    edge_index = self._get_edge_index(self.image_size//self.downsample_ratio)
                    # y = torch.zeros((self.image_size//self.downsample_ratio*self.image_size//self.downsample_ratio,1), dtype=torch.float)
                    y = mask.bool()
                    graph = Data(x=x, edge_index=edge_index,y=y)
                    # print(graph)
                    outlier_scores =self.methods.decision_function(graph)
                    y_pre.extend(outlier_scores)
                    img_y_pre.extend(np.mean(outlier_scores))
                    y_true.extend(graph.y.numpy())
                    img_y_true.extend(label[0])
            total_auc = roc_auc_score(y_true, y_pre) 
            total_ap = average_precision_score(y_true, y_pre)
            print(f'AUC Score: {total_auc:.3f}') 
            print(f'AP Score: {total_ap:.3f}')
                
        return total_auc,total_ap

        # for method_name, method in self.methods.items():
        #     method.calculate_metrics()
        #     image_rocaucs[method_name] = round(method.image_rocauc, 3)
        #     pixel_rocaucs[method_name] = round(method.pixel_rocauc, 3)
        #     au_pros[method_name] = round(method.au_pro, 3)
        #     print(
        #         f'Class: {class_name}, {method_name} Image ROCAUC: {method.image_rocauc:.3f}, {method_name} Pixel ROCAUC: {method.pixel_rocauc:.3f}, {method_name} AU-PRO: {method.au_pro:.3f}')
        # return image_rocaucs, pixel_rocaucs, au_pros















    def _train_anomaly_detector(self,model,graph):
        return model.fit(graph)

    def _get_edge_index(self,num=56):
        edge_index = []
        for i in range(num):
            for j in range(num):
                if (i>0):
                    edge_index.append([i*num+j,(i-1)*num+j])
                if (i<num-1):
                    edge_index.append([i*num+j,(i+1)*num+j])
                if (j>0):
                    edge_index.append([i*num+j,i*num+j-1])
                if (j<num-1):
                    edge_index.append([i*num+j,i*num+j+1])
        edge_index = np.array(edge_index).T
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        return edge_index

