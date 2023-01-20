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
        self.downsample_ratio = 8
        self.avg_pooling = nn.AvgPool2d(kernel_size=self.downsample_ratio)
        self.k_neig = 6
        self.hid_dim = 12
        self.num_layers = 6
        self.dropout = 0.5
        self.act = F.relu
        self.eta = 0.5
        self.lr = 1e-3
        self.weight_decay = 0.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.margin=.5
        self.margin_loss_func = torch.nn.MarginRankingLoss(margin=self.margin)
        self.r = 0.2
        self.m = 50
        self.k = 50
        self.f = 10
        self.alpha = 0.5
        self.model = CONAD_Base(in_dim=self.downsample_ratio*self.downsample_ratio*3,
                                            hid_dim=self.hid_dim,
                                            num_layers=self.num_layers,
                                            dropout=self.dropout,
                                            act=self.act).to(self.device)
        self.img_pred_list = [] # list<numpy>
        self.img_gt_list = [] # list<numpy>
        self.pixel_pred_list = [] # list<numpy(m,n)>
        self.pixel_gt_list = [] # list<numpy(m,n)>
        
    def fit(self, class_name):
        
        train_loader = get_data_loader("train", class_name=class_name, img_size=self.image_size)
        k = 0
        # self.model = CONAD_Base(in_dim=self.downsample_ratio*self.downsample_ratio*3,
        #                                     hid_dim=self.hid_dim,
        #                                     num_layers=self.num_layers,
        #                                     dropout=self.dropout,
        #                                     act=self.act).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        
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

            for x in Data_xyz:
                edge_index = util.construct_H_with_KNN(x, k_neig = self.k_neig)
                y = torch.zeros((self.image_size//self.downsample_ratio*self.image_size//self.downsample_ratio,1), dtype=torch.float)
                y = y.bool()
                graph = Data(x=x, edge_index=edge_index,y=y)
                print("graph:",graph)
                
                # # g = to_networkx(graph,to_undirected=True)
                # # visualize_graph(g,graph.y)
                # # batch_size = sampled_data.batch_size
                # # node_idx = sampled_data.node_idx
                
                # x, s, edge_index = self.process_graph(graph)
                # print("x:",x.shape)
                # print("s:",s.shape)
                # print("edge_index:",edge_index.shape)
                # # x_aug, edge_index_aug, label_aug = self._data_augmentation(x, s)
                # # h_aug = self.model.embed(x_aug, edge_index_aug)
                # # h = self.model.embed(x, edge_index)

                

                # # margin_loss = self.margin_loss_func(h, h, h_aug) * label_aug
                # # margin_loss = torch.mean(margin_loss)
                # # x_, s_ = self.model.reconstruct(h, edge_index)
                # x_, s_ = self.model(x, edge_index)
                # # print("x:",x_.shape)
                # # print("s:",s_.shape)
                # score = self.loss_func(x,x_,s_,s_)

                # # loss = self.eta * torch.mean(score) + (1 - self.eta) * margin_loss 
                # loss = torch.mean(score)
                # print(loss.item)
                # # print("loss:",loss)

                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()


        # print(f"class: {class_name} Loss: {self.model.loss}")
        # torch.save(self.methods.model, '/ssd2/m3lab/usrs/crt/3D-ADS/model_saved/tire_gnn_10_1_7.pt')
        
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
                # mask = self.avg_pooling(mask)
                # mask = mask.view(-1,1)
                # mask = torch.where(mask > 0.5, 1., .0)
                mask = mask.reshape(224,224)
                # print("mask:",mask.shape)
                mask = mask.cpu().numpy()
                # print("mask:",mask.shape)

                x = []
                for i in range(self.downsample_ratio):
                    for j in range(self.downsample_ratio):
                        x1 = Data_xyz[:,:,i::self.downsample_ratio,j::self.downsample_ratio]
                        x.append(x1)

                Data_xyz = torch.cat(x,dim=1)
                Data_xyz = Data_xyz.permute(0,2,3,1)
                Data_xyz = Data_xyz.view(1,-1,3*self.downsample_ratio*self.downsample_ratio)
                threshold = 1.5
                for x in Data_xyz:
                    edge_index = util.construct_H_with_KNN(x, k_neig = self.k_neig)
                    # edge_index = self._get_edge_index(self.image_size//self.downsample_ratio)
                    # y = mask.bool()
                    graph = Data(x=x, edge_index=edge_index)
                    x, s, edge_index = self.process_graph(graph)
                    x_, s_ = self.model(x, edge_index)
                    print("x_",x_.shape)
                    x = x.cpu().numpy()
                    x_ = x_.cpu().numpy()
                    result_map = np.sum(np.abs(x-x_),axis = 1)
                    print("result_map",result_map.shape)
                    result_map = result_map.reshape(int(self.image_size/self.downsample_ratio),int(self.image_size/self.downsample_ratio),-1)
                    print("result_map",result_map.shape)
                    map_resized = cv.resize(result_map,(224,224))
                    map_cv = gaussian_filter(map_resized, sigma=4)
                    print("map_cv",map_cv)
                    print("map_cv",map_cv.shape)

                    y = map_cv.copy()
                    y[map_cv/np.mean(map_cv)>threshold] = 1
                    # print("y",y)
                    y[map_cv/np.mean(map_cv)<=threshold] = 0
                    # print("y",y)
                    self.pixel_gt_list.extend(mask.flatten())
                    self.pixel_pred_list.extend(y.flatten())
                    self.img_gt_list.append(label.cpu().numpy()[0])

                    self.img_pred_list.append(np.max(y))
                    # graph = Data(x=x, edge_index=edge_index,y=y)
                    # g = to_networkx(graph,to_undirected=True)
                    # visualize_graph(g,graph.y)
                    # print(graph)
                    # outlier_scores =self.methods.decision_function(graph)
                    # y_pre.extend(outlier_scores)
                    # img_y_pre.extend(np.mean(outlier_scores))
                    # y_true.extend(graph.y.numpy())
                    # img_y_true.extend(label[0]) 
            # print("self.pixel_gt_list",self.pixel_gt_list==0)
            # print("self.pixel_gt_list",self.pixel_gt_list==1)
            # print("self.pixel_pred_list",self.pixel_pred_list==0)
            # print("self.pixel_pred_list",self.pixel_pred_list==1)
            # print("self.pixel_gt_list",self.pixel_gt_list)
            # print("self.pixel_gt_list",len(self.pixel_gt_list))
            # print("self.pixel_pred_list",self.pixel_pred_list)
            # print("self.pixel_pred_list",len(self.pixel_pred_list))
            pixel_auroc = roc_auc_score(self.pixel_gt_list, self.pixel_pred_list)
            img_auroc = roc_auc_score(self.img_gt_list, self.img_pred_list)
            pixel_ap = average_precision_score(self.pixel_gt_list, self.pixel_pred_list)
            img_ap = average_precision_score(self.img_gt_list, self.img_pred_list)
            # total_auc = roc_auc_score(y_true, y_pre) 
            # total_ap = average_precision_score(y_true, y_pre)
            # print(f'AUC Score: {total_auc:.3f}') 
            # print(f'AP Score: {total_ap:.3f}')
                
        return pixel_auroc, img_auroc, pixel_ap, img_ap

        # for method_name, method in self.methods.items():
        #     method.calculate_metrics()
        #     image_rocaucs[method_name] = round(method.image_rocauc, 3)
        #     pixel_rocaucs[method_name] = round(method.pixel_rocauc, 3)
        #     au_pros[method_name] = round(method.au_pro, 3)
        #     print(
        #         f'Class: {class_name}, {method_name} Image ROCAUC: {method.image_rocauc:.3f}, {method_name} Pixel ROCAUC: {method.pixel_rocauc:.3f}, {method_name} AU-PRO: {method.au_pro:.3f}')
        # return image_rocaucs, pixel_rocaucs, au_pros
    def loss_func(self, x, x_, s, s_):
        # attribute reconstruction loss
        diff_attribute = torch.pow(x - x_, 2)
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        # structure reconstruction loss
        diff_structure = torch.pow(s - s_, 2)
        structure_errors = torch.sqrt(torch.sum(diff_structure, 1))

        score = self.alpha * attribute_errors \
                + (1 - self.alpha) * structure_errors
        return score
    def _data_augmentation(self, x, adj):
        """
        Data augmentation on the input graph. Four types of
        pseudo anomalies will be injected:
            Attribute, deviated
            Attribute, disproportionate
            Structure, high-degree
            Structure, outlying
        
        Parameters
        -----------
        x : note attribute matrix
        adj : dense adjacency matrix

        Returns
        -------
        feat_aug, adj_aug, label_aug : augmented
            attribute matrix, adjacency matrix, and
            pseudo anomaly label to train contrastive
            graph representations
        """
        rate = self.r
        num_added_edge = self.m
        surround = self.k
        scale_factor = self.f

        adj_aug, feat_aug = deepcopy(adj), deepcopy(x)
        num_nodes = adj_aug.shape[0]
        label_aug = torch.zeros(num_nodes, dtype=torch.int32)

        prob = torch.rand(num_nodes)
        label_aug[prob < rate] = 1

        # high-degree
        n_hd = torch.sum(prob < rate / 4)
        edges_mask = torch.rand(n_hd, num_nodes) < num_added_edge / num_nodes
        edges_mask = edges_mask.to(self.device)
        adj_aug[prob <= rate / 4, :] = edges_mask.float()
        adj_aug[:, prob <= rate / 4] = edges_mask.float().T

        # outlying
        ol_mask = torch.logical_and(rate / 4 <= prob, prob < rate / 2)
        adj_aug[ol_mask, :] = 0
        adj_aug[:, ol_mask] = 0

        # deviated
        dv_mask = torch.logical_and(rate / 2 <= prob, prob < rate * 3 / 4)
        feat_c = feat_aug[torch.randperm(num_nodes)[:surround]]
        ds = torch.cdist(feat_aug[dv_mask], feat_c)
        feat_aug[dv_mask] = feat_c[torch.argmax(ds, 1)]

        # disproportionate
        mul_mask = torch.logical_and(rate * 3 / 4 <= prob, prob < rate * 7 / 8)
        div_mask = rate * 7 / 8 <= prob
        feat_aug[mul_mask] *= scale_factor
        feat_aug[div_mask] /= scale_factor

        edge_index_aug = dense_to_sparse(adj_aug)[0].to(self.device)
        feat_aug = feat_aug.to(self.device)
        label_aug = label_aug.to(self.device)
        return feat_aug, edge_index_aug, label_aug

    def process_graph(self, G):
        """
        Process the raw PyG data object into a tuple of sub data
        objects needed for the model.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        x : torch.Tensor
            Attribute (feature) of nodes.
        s : torch.Tensor
            Adjacency matrix of the graph.
        edge_index : torch.Tensor
            Edge list of the graph.
        """
        s = to_dense_adj(G.edge_index)[0].to(self.device)
        edge_index = G.edge_index.to(self.device)
        x = G.x.to(self.device)

        return x, s, edge_index



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

def to_dense_adj(edge_index, batch=None, edge_attr=None, max_num_nodes=None):
    r"""Converts batched sparse adjacency matrices given by edge indices and
    edge attributes to a single dense batched adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = num_nodes.max().item()

    elif idx1.max() >= max_num_nodes or idx2.max() >= max_num_nodes:
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = torch.ones(idx0.numel(), device=edge_index.device)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.size())[1:]
    adj = torch.zeros(size, dtype=edge_attr.dtype, device=edge_index.device)

    flattened_size = batch_size * max_num_nodes * max_num_nodes
    adj = adj.view([flattened_size] + list(adj.size())[3:])
    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    scatter(edge_attr, idx, dim=0, out=adj, reduce='add')
    adj = adj.view(size)

    return adj

def dense_to_sparse(adj):
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.

    Args:
        adj (Tensor): The dense adjacency matrix.
     :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    index = adj.nonzero(as_tuple=True)
    edge_attr = adj[index]

    if len(index) == 3:
        batch = index[0] * adj.size(-1)
        index = (batch + index[1], batch + index[2])

    return torch.stack(index, dim=0), edge_attr