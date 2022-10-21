# Latent Outlier Exposure for Anomaly Detection with Contaminated Data
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from scipy import io
from .utils import *
import torch
import numpy as np

def CIFAR10_feat(normal_class,root='DATA/cifar10_features/',contamination_rate=0.0):
    trainset = torch.load(root+'trainset_2048.pt')
    train_data,train_targets = trainset
    testset = torch.load(root+'testset_2048.pt')
    test_data,test_targets = testset
    test_labels = np.ones_like(test_targets)
    test_labels[test_targets==normal_class]=0
    
    train_clean = train_data[train_targets==normal_class]
    train_contamination = train_data[train_targets!=normal_class]
    num_clean = train_clean.shape[0]
    num_contamination = int(contamination_rate/(1-contamination_rate)*num_clean)

    idx_contamination = np.random.choice(np.arange(train_contamination.shape[0]),num_contamination,replace=False)
    train_contamination = train_contamination[idx_contamination]    
    train_data = torch.cat((train_clean,train_contamination),0)
        
    train_labels = np.zeros(train_data.shape[0])
    train_labels[num_clean:]=1

    return train_data,train_labels,test_data,test_labels


def FMNIST_feat(normal_class, root='DATA/fmnist_features/',contamination_rate=0.0):
    trainset = torch.load(root + 'trainset_2048.pt')
    train_data, train_targets = trainset
    testset = torch.load(root + 'testset_2048.pt')
    test_data, test_targets = testset
    test_labels = np.ones_like(test_targets)
    test_labels[test_targets == normal_class] = 0

    train_clean = train_data[train_targets == normal_class]
    train_contamination = train_data[train_targets != normal_class]
    num_clean = train_clean.shape[0]
    num_contamination = int(contamination_rate / (1 - contamination_rate) * num_clean)

    idx_contamination = np.random.choice(np.arange(train_contamination.shape[0]), num_contamination, replace=False)
    train_contamination = train_contamination[idx_contamination]
    train_data = torch.cat((train_clean, train_contamination), 0)

    train_labels = np.zeros(train_data.shape[0])
    train_labels[num_clean:] = 1

    return train_data, train_labels, test_data, test_labels

def synthetic_contamination(norm,anorm,contamination_rate):
    num_clean = norm.shape[0]
    num_contamination = int(contamination_rate/(1-contamination_rate)*num_clean)

    try:
        idx_contamination = np.random.choice(np.arange(anorm.shape[0]),num_contamination,replace=False)
    except:
        idx_contamination = np.random.choice(np.arange(anorm.shape[0]),num_contamination,replace=True)
    train_contamination = anorm[idx_contamination]
    train_contamination = train_contamination + np.random.randn(*train_contamination.shape)*np.std(anorm,0,keepdims=True)
    train_data = np.concatenate([norm,train_contamination],0)
    train_labels = np.zeros(train_data.shape[0])
    train_labels[num_clean:]=1
    return train_data,train_labels

def Thyroid_data(contamination_rate):
    data = io.loadmat("DATA/thyroid.mat")
    samples = data['X']  # 3772
    labels = ((data['y']).astype(np.int32)).reshape(-1)

    inliers = samples[labels == 0]  # 3679 norm
    outliers = samples[labels == 1]  # 93 anom

    num_split = len(inliers) // 2
    train_norm = inliers[:num_split]  # 1839 train
    test_data = np.concatenate([inliers[num_split:], outliers], 0)
    test_label = np.zeros(test_data.shape[0])
    test_label[num_split:] = 1

    train, train_label = synthetic_contamination(train_norm, outliers, contamination_rate)
    return train, train_label, test_data, test_label


def Arrhythmia_data(contamination_rate):
    data = io.loadmat("DATA/arrhythmia.mat")
    samples = data['X']  # 518
    labels = ((data['y']).astype(np.int32)).reshape(-1)

    inliers = samples[labels == 0]  # 3679 norm
    outliers = samples[labels == 1]  # 93 anom

    num_split = len(inliers) // 2
    train_norm = inliers[:num_split]  # 1839 train
    test_data = np.concatenate([inliers[num_split:], outliers], 0)
    test_label = np.zeros(test_data.shape[0])
    test_label[num_split:] = 1

    train, train_label = synthetic_contamination(train_norm, outliers, contamination_rate)
    return train, train_label, test_data, test_label


def load_data(data_name,cls,contamination_rate=0.0):

    ## normal data with label 0, anomalies with label 1

    if data_name == 'cifar10_feat':
        train, train_label, test, test_label = CIFAR10_feat(cls,contamination_rate=contamination_rate)
    elif data_name == 'fmnist_feat':
        train, train_label, test, test_label = FMNIST_feat(cls, contamination_rate=contamination_rate)
    elif data_name == 'thyroid':
        train, train_label, test, test_label = Thyroid_data(contamination_rate)
    elif data_name == 'arrhythmia':
        train, train_label, test, test_label = Arrhythmia_data(contamination_rate)

    trainset = CustomDataset(train,train_label)
    testset = CustomDataset(test,test_label)
    return trainset,testset,testset






