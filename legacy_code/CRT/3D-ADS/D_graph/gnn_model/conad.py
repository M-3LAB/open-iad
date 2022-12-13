# -*- coding: utf-8 -*-
"""Contrastive Attributed Network Anomaly Detection
with Data Augmentation (CONAD)"""
# Author: Zhiming Xu <zhimng.xu@gmail.com>
# License: BSD 2 clause

import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from sklearn.utils.validation import check_is_fitted

from . import BaseDetector
from .basic_nn import GCN
# from ..metrics import eval_roc_auc
# from ..utils import validate_device


class CONAD(BaseDetector):
    """
    CONAD (Contrastive Attributed Network Anomaly Detection) is an
    anomaly detector consisting of a shared graph
    convolutional encoder, a structure reconstruction decoder, and an
    attribute reconstruction decoder. The model is trained with both
    contrastive loss and structure/attribute reconstruction loss.
    The reconstruction mean square error of the decoders are defined
    as structure anomaly score and attribute anomaly score, respectively.

    See :cite:`xu2022contrastive` for details.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. A half (ceil) of the layers
        are for the encoder, the other half (floor) of the layers are
        for decoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.3``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    alpha : float, optional
        Loss balance weight for attribute and structure. ``None`` for
        balancing by standard deviation. Default: ``None``.
    eta : float, optional
        Loss balance weight for contrastive and reconstruction.
        Default: ``0.5``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.005``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``5``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``0``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    r : float, optional
        The rate of augmented anomalies. Default: ``.2``.
    m : int, optional
        For densely connected nodes, the number of
        edges to add. Default: ``50``.
    k : int, optional
        same as ``k`` in ``pygod.generator.gen_contextual_outliers``.
        Default: ``50``.
    f : int, optional
        For disproportionate nodes, the scale factor applied
        on their attribute value. Default: ``10``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Default: ``False``.

    Examples
    --------
    >>> from pygod.models import CONAD
    >>> model = CONAD()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.3,
                 weight_decay=0.,
                 act=F.relu,
                 alpha=None,
                 eta=.5,
                 contamination=0.1,
                 lr=1e-3,
                 epoch=5,
                 gpu=0,
                 batch_size=0,
                 num_neigh=-1,
                 margin=.5,
                 r=.2,
                 m=50,
                 k=50,
                 f=10,
                 verbose=False):
        super(CONAD, self).__init__(contamination=contamination)

        # model param
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.alpha = alpha
        self.eta = eta

        # training param
        self.lr = lr
        self.epoch = epoch
        # self.device = validate_device(gpu)
        self.batch_size = batch_size
        self.num_neigh = num_neigh

        self.margin_loss_func = torch.nn.MarginRankingLoss(margin=margin)
        # other param
        self.verbose = verbose
        self.r = r
        self.m = m
        self.k = k
        self.f = f
        self.model = None
        self.train_first = True
        self.loss = 0


    def fit(self, G, y_true=None):
        """
        Fit detector with input data.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.
        y_true : numpy.ndarray, optional
            The optional outlier ground truth labels used to monitor
            the training progress. They are not used to optimize the
            unsupervised model. Default: ``None``.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        G.node_idx = torch.arange(G.x.shape[0])

        # automated balancing by std
        if self.alpha is None:
            adj = to_dense_adj(G.edge_index)[0]
            self.alpha = torch.std(adj).detach() / \
                         (torch.std(G.x).detach() + torch.std(adj).detach())
            adj = None

        if self.batch_size == 0:
            self.batch_size = G.x.shape[0]
        loader = NeighborLoader(G,
                                [self.num_neigh] * self.num_layers,
                                batch_size=self.batch_size)
        if self.train_first == True:
            self.model = CONAD_Base(in_dim=G.x.shape[1],
                                    hid_dim=self.hid_dim,
                                    num_layers=self.num_layers,
                                    dropout=self.dropout,
                                    act=self.act).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        self.model.train()
        decision_scores = np.zeros(G.x.shape[0])
        for epoch in range(self.epoch):
            epoch_loss = 0
            for sampled_data in loader:
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.node_idx
                x, s, edge_index = self.process_graph(sampled_data)
                # x:特征 s:邻接矩阵 edge_index 边
                # print("x:",x)
                # print("x:",x.shape)
                # print("s:",s)
                # print("s:",s.shape)
                # print("edge_index:",edge_index)
                # print("edge_index:",edge_index.shape)
                # generate augmented graph
                x_aug, edge_index_aug, label_aug = \
                    self._data_augmentation(x, s)

                # print("x_aug:",x_aug)
                # print("x_aug:",x_aug.shape)
                # print("edge_index_aug:",edge_index_aug)
                # print("edge_index_aug:",edge_index_aug.shape)
                # print("label_aug:",label_aug)
                # print("label_aug:",label_aug.shape)

                h_aug = self.model.embed(x_aug, edge_index_aug)
                # print("h_aug:",h_aug)
                # print("h_aug:",h_aug.shape)
                h = self.model.embed(x, edge_index)
                # print("h:",h)
                # print("h:",h.shape)

                margin_loss = self.margin_loss_func(h, h, h_aug) * label_aug
                # print("margin_loss:",margin_loss)
                margin_loss = torch.mean(margin_loss)
                # print("margin_loss:",margin_loss)


                x_, s_ = self.model.reconstruct(h, edge_index)
                # print("x_:",x_)
                # print("s_:",s_)
                # print("batch_size:",batch_size)
                score = self.loss_func(x[:batch_size],
                                       x_[:batch_size],
                                       s[:batch_size],
                                       s_[:batch_size])
                # print("self.eta:",self.eta)
                # print("margin_loss:",margin_loss)
                # print("score:",score)
                loss = self.eta * torch.mean(score) + \
                       (1 - self.eta) * margin_loss                
                decision_scores[node_idx[:batch_size]] = score.detach() \
                    .cpu().numpy()
                epoch_loss += loss.item() * batch_size

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.loss =  epoch_loss / G.x.shape[0]
            if self.verbose:
                # print("epoch_loss:",epoch_loss)
                # print("G.x.shape[0]:",G.x.shape[0])
                print("Epoch {:04d}: Loss {:.4f}"
                      .format(epoch, epoch_loss / G.x.shape[0]), end='')
                # if y_true is not None:
                #     auc = eval_roc_auc(y_true, decision_scores)
                #     print(" | AUC {:.4f}".format(auc), end='')
                # print()
            # torch.save(self.model, '/ssd2/m3lab/usrs/crt/3D-ADS/model_saved/gnn_30_1_8.pt')


        self.decision_scores_ = decision_scores
        self._process_decision_scores()
        self.train_first = False
        return self

    def decision_function(self, G):
        """
        Predict raw anomaly score using the fitted detector. Outliers
        are assigned with larger anomaly scores.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        outlier_scores : numpy.ndarray
            The anomaly score of shape :math:`N`.
        """
        check_is_fitted(self, ['model'])
        G.node_idx = torch.arange(G.x.shape[0])
        G.s = to_dense_adj(G.edge_index)[0]

        loader = NeighborLoader(G,
                                [self.num_neigh] * self.num_layers,
                                batch_size=self.batch_size)

        self.model.eval()
        outlier_scores = np.zeros(G.x.shape[0])
        for sampled_data in loader:
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.node_idx

            x, s, edge_index = self.process_graph(sampled_data)

            x_, s_ = self.model(x, edge_index)
            score = self.loss_func(x[:batch_size],
                                   x_[:batch_size],
                                   s[:batch_size],
                                   s_[:batch_size])

            outlier_scores[node_idx[:batch_size]] = score.detach() \
                .cpu().numpy()
        return outlier_scores

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


class CONAD_Base(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_layers,
                 dropout,
                 act):
        super(CONAD_Base, self).__init__()

        decoder_layers = int(num_layers / 2)
        encoder_layers = num_layers - decoder_layers

        self.shared_encoder = GCN(in_channels=in_dim,
                                  hidden_channels=hid_dim,
                                  num_layers=encoder_layers,
                                  out_channels=hid_dim,
                                  dropout=dropout,
                                  act=act)

        self.attr_decoder = GCN(in_channels=hid_dim,
                                hidden_channels=hid_dim,
                                num_layers=decoder_layers,
                                out_channels=in_dim,
                                dropout=dropout,
                                act=act)

        self.struct_decoder = GCN(in_channels=hid_dim,
                                  hidden_channels=hid_dim,
                                  num_layers=decoder_layers - 1,
                                  out_channels=in_dim,
                                  dropout=dropout,
                                  act=act)

    def embed(self, x, edge_index):
        h = self.shared_encoder(x, edge_index)
        return h

    def reconstruct(self, h, edge_index):
        # decode attribute matrix
        x_ = self.attr_decoder(h, edge_index)
        # decode structure matrix
        h_ = self.struct_decoder(h, edge_index)

        s_ = h_ @ h_.T
        return x_, s_

    def forward(self, x, edge_index):
        # encode
        h = self.embed(x, edge_index)
        # reconstruct
        x_, s_ = self.reconstruct(h, edge_index)
        return x_, s_
