import torch
from models._patchcore.kcenter_greedy import KCenterGreedy 
import cv2
import torch.nn.functional as F
import numpy as np
from sklearn.random_projection import SparseRandomProjection
import faiss
import math
from scipy.ndimage import gaussian_filter
from arch_base.base import ModelBase
from tools.utils import *
import os

__all__ = ['GraphCore']

class GraphCore(ModelBase):

    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        super(GraphCore, self).__init__(config, device)

        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
        self.model = self.net.model
        self.model.to(self.device)

        self.features = []
        self.random_projector = SparseRandomProjection(n_components='auto', eps=0.9)
        self.embedding_coreset = np.array([])
        self.embedding_path = self.file_path + '/embed'
        create_folders(self.embedding_path)

        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def get_layer_features(self):
        pass
    
    def train_model(self, train_loader, task_id, inf=''):
        pass

    def prediction(self, valid_loader, task_id):
        pass