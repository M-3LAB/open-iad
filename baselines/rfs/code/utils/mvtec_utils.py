import numpy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from kpdataset.mvteckp import mvtec_data
from torch.distributions import multivariate_normal as mm
import cv2
from torch.utils.data import TensorDataset
import scipy.io as scio

def d2_mat_loader(path,max_card=None):
    sample =scio.loadmat(path)
    sample=sample['descriptors']
    if max_card is not None:
        sample=sample
    else:
        if sample.ndim==0:
            print('loading image with no keypoints')
            sample=torch.from_numpy(np.zeros([1,512],dtype=float))
        else:
            sample=torch.from_numpy(sample)
    return sample


def get_cardinality(your_dataloader):
    card_train_data = []
    for i, batch in enumerate(your_dataloader):
        card_train_data.append(batch[2])
    card_train_data
    train_cardinality=card_train_data
    max_card=torch.max(torch.tensor(train_cardinality))
    return train_cardinality,max_card

def get_cardinality_dataset(your_dataloader):
    card_train_data = []
    data_tensor=[]
    data_tensor_label=[]
    for i, batch in enumerate(your_dataloader):
        card_train_data.append(batch[2])
        data_tensor.append(batch[0].squeeze(dim=0))
        data_tensor_label.append(torch.zeros(batch[2]))

    data_tensor = torch.cat(data_tensor)
    data_tensor_label=torch.cat(data_tensor_label)
    card_train_data
    train_cardinality=card_train_data
    max_card=torch.max(torch.tensor(train_cardinality))
    return train_cardinality,max_card,data_tensor,data_tensor_label
def get_pts_mean(your_dataloader):
    pts_prob_mean = []
    for i, batch in enumerate(your_dataloader):
        sample_pts,_,_= batch
        sample_pts=sample_pts.squeeze(dim=0)
        pts_mean = torch.tensor([sample_pts[i][2] for i in range(sample_pts.shape[0])]).mean().unsqueeze(dim=0)
        pts_prob_mean.append(pts_mean)
    pts_prob_mean=torch.tensor(pts_prob_mean)
    pts_mean=pts_prob_mean.mean()
    return pts_mean


def get_mvtec(args,train_only=True,with_vaild=True):
    """Returning train and test dataloaders. This version ensure each batch carry keypoints rather than images"""
    data_dir = args.data_dir
    feat_type=args.feat_type
    #load the keypoints for first time to get the max lengths
    if feat_type=='lf_net':
        feat_loader=lf_npy_loader
        ext='.npz'
    if feat_type=='d2_net' or feat_type=='d2_net2' or feat_type=='d2_net4' or feat_type=='d2_net6':
        feat_loader = d2_mat_loader
        ext = '.mat'
    if feat_type=='r2d2':
        feat_loader = r2d2_loader
        ext = '.r2d2'
    if feat_type=='sp':
        feat_loader=sp_npy_loader
        ext='.npz'
    if feat_type=='sp-sift':
        feat_loader=sp_npy_loader_sift
        ext='.npz'




    # load image keypoint dataset using batch_size one to find the max length (return keypoints of each image)
    train_img_kp=mvtec_data(root=data_dir,loader=feat_loader,extensions=ext,classes=['train'],only_card=False)
    ##dataloader using batch_size=1 to get the max cardinlaity for purpose of padding
    trainloader_set = torch.utils.data.DataLoader(train_img_kp, batch_size=1, shuffle=False)
    # obtain the max samples and min samples to normalise
    max_sample, min_sample = get_max_sp(trainloader_set)
    #lload image normalized or not normalized image keypoit dataset, then load keypoint dataset class
    if args.normalise_data==1:

        train_img_kp = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False,sample_max=max_sample,
                               sample_min=min_sample)
        trainloader_set = torch.utils.data.DataLoader(train_img_kp, batch_size=1, shuffle=False)
        # get the max lenght of features, keppoint datasetmatrix
        train_cardinality, max_card_train, dataset_matrix, dataset_matrix_label = get_cardinality_dataset(trainloader_set)
        #convert keypoints datasetmatrix into keppoint datset classs


    else:
        train_img_kp = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False)
        trainloader_set = torch.utils.data.DataLoader(train_img_kp, batch_size=1, shuffle=False)
        train_cardinality, max_card_train, dataset_matrix, dataset_matrix_label = get_cardinality_dataset(trainloader_set)
        # convert keypoints datasetmatrix into keppoint datset classs

    my_train_kp_dataset = TensorDataset(dataset_matrix, dataset_matrix_label)

    # splite the traning dataset into validation and testing using 20% pecent of normal samples
    if with_vaild:
        train_set, val_set = torch.utils.data.random_split(my_train_kp_dataset, [int(len(my_train_kp_dataset)-(int(0.2*len(my_train_kp_dataset)))), int(0.2*len(my_train_kp_dataset))])
    #the padded train dataloader
    else:
        train_set=my_train_kp_dataset

    if args.batch_size=='full':
        dataloader_train = DataLoader(train_set, batch_size=len(train_set), shuffle=True, num_workers=0)
        if with_vaild:
            dataloader_vald = DataLoader(val_set, batch_size=len(val_set), shuffle=True, num_workers=0)
    elif args.batch_size == 'half':
        dataloader_train = DataLoader(train_set, batch_size=(len(train_set)//2), shuffle=True, num_workers=0)
        if with_vaild:
            dataloader_vald = DataLoader(val_set,batch_size=(len(val_set)//2), shuffle=True, num_workers=0)

    else:
        dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        if with_vaild:
            dataloader_vald = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=0)


    # the test with padding, the test set with no padding
    if args.normalise_data == 1:
        test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False,sample_max=max_sample,
                               sample_min=min_sample)
    else:
        test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False)

    testloader_set = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    # the padded test set using max_test

    train_card = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=True)
    card_trainer = torch.utils.data.DataLoader(train_card, batch_size=len(train_img_kp), shuffle=False)

    # this dataloader will be used for evaluation
    trainloader_eval = torch.utils.data.DataLoader(train_img_kp, batch_size=1, shuffle=False)
    #trainloader_set = torch.utils.data.DataLoader(train_set_2, batch_size=1, shuffle=False)

    #return dataloader_train, dataloader_test, trainloader_set, testloader_set, trainloader_eval
    if train_only:

        if with_vaild:
            print('return train_loader, vailidation loader, card loader')
            return dataloader_train, dataloader_vald, card_trainer
        print('return train_loader, card loader')
        return dataloader_train, card_trainer
    

    return dataloader_train, testloader_set, card_trainer, testloader_set, trainloader_eval

def get_mvtec_datamatrix(args,train_only=True,with_vaild=True):
    """Returning train datamatrix and test datamatrix"""
    data_dir = args.data_dir
    feat_type=args.feat_type
    #load the keypoints for first time to get the max lengths
    if feat_type=='lf_net':
        feat_loader=lf_npy_loader
        ext='.npz'
    if feat_type=='d2_net' or feat_type=='d2_net4':
        feat_loader = d2_mat_loader
        ext = '.mat'
    if feat_type=='r2d2':
        feat_loader = r2d2_loader
        ext = '.r2d2'
    if feat_type=='sp':
        feat_loader=sp_npy_loader
        ext='.npz'
    if feat_type=='sp-sift':
        feat_loader=sp_npy_loader_sift
        ext='.npz'

    # load image keypoint dataset using batch_size one to find the max length (return keypoints of each image)
    train_img_kp=mvtec_data(root=data_dir,loader=feat_loader,extensions=ext,classes=['train'],only_card=False)
    ##dataloader using batch_size=1 to get the max cardinlaity for purpose of padding
    trainloader_set = torch.utils.data.DataLoader(train_img_kp, batch_size=1, shuffle=False)
    # obtain the max samples and min samples to normalise
    max_sample, min_sample = get_max_sp(trainloader_set)
    #lload image normalized or not normalized image keypoit dataset, then load keypoint dataset class
    if args.normalise_data==1:

        train_img_kp = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False,sample_max=max_sample,
                               sample_min=min_sample)
        trainloader_set = torch.utils.data.DataLoader(train_img_kp, batch_size=1, shuffle=False)
        # get the max lenght of features, keppoint datasetmatrix
        train_cardinality, max_card_train, dataset_matrix, dataset_matrix_label = get_cardinality_dataset(trainloader_set)
        #convert keypoints datasetmatrix into keppoint datset classs


    else:
        train_img_kp = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False)
        trainloader_set = torch.utils.data.DataLoader(train_img_kp, batch_size=1, shuffle=False)
        train_cardinality, max_card_train, dataset_matrix, dataset_matrix_label = get_cardinality_dataset(trainloader_set)
        # convert keypoints datasetmatrix into keppoint datset classs

    my_train_kp_dataset = TensorDataset(dataset_matrix, dataset_matrix_label)

    # splite the traning dataset into validation and testing using 20% pecent of normal samples
    if with_vaild:
        train_set, val_set = torch.utils.data.random_split(my_train_kp_dataset, [int(len(my_train_kp_dataset)-(int(0.2*len(my_train_kp_dataset)))), int(0.2*len(my_train_kp_dataset))])
    #the padded train dataloader
    else:
        train_set=my_train_kp_dataset



    # the test with padding, the test set with no padding
    if args.normalise_data == 1:
        test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False,sample_max=max_sample,
                               sample_min=min_sample)
    else:
        test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False)

    testloader_set = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    # the padded test set using max_test

    train_card = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=True)
    card_trainer = torch.utils.data.DataLoader(train_card, batch_size=len(train_img_kp), shuffle=False)

    # this dataloader will be used for evaluation
    trainloader_eval = torch.utils.data.DataLoader(train_img_kp, batch_size=1, shuffle=False)
    #trainloader_set = torch.utils.data.DataLoader(train_set_2, batch_size=1, shuffle=False)

    #return dataloader_train, dataloader_test, trainloader_set, testloader_set, trainloader_eval
    if train_only:
        return dataset_matrix, card_trainer

    return dataset_matrix, testloader_set, card_trainer, testloader_set, trainloader_eval
