from data.mvtec3d import get_data_loader
import torch
from tqdm import tqdm
import cv2
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
class_name="bagel"
train_loader = get_data_loader("test", class_name=class_name, img_size=224)
k = 1
sresize = torch.nn.AdaptiveAvgPool2d((28, 28))
saverage = torch.nn.AvgPool2d(3, stride=1)

for sample,gt,_ in tqdm(train_loader, desc=f'Extracting train features for class {class_name}'):
    if k == 1:
        # print(sample)
        a = 1
        # rgb_feature_maps = sample[0]
        # rgb_resized_maps = [sresize(saverage(fmap)) for fmap in rgb_feature_maps]
        # rgb_resized_maps = [rgb_resized_maps[0],rgb_resized_maps[0]]
        # print("rgb_resized_maps:",rgb_resized_maps[0].shape)
        # print("rgb_resized_maps:",len(rgb_resized_maps))
        # print()
        # rgb_patch = torch.cat(rgb_resized_maps, 1)
        # print("rgb_patch:",rgb_patch.shape)

        # rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        # print("rgb_patch:",rgb_patch.shape)
        # patch_lib = [rgb_patch,rgb_patch,rgb_patch]
        # patch_lib = torch.cat(patch_lib, 0)
        # print("patch_lib:",patch_lib.shape)
        # data_train = sample[1]
        # y = data_train.numpy();
        # print(y)
        # print(np.size(y))
        # np.save("tiff_xyz.npy", y)
        # data_train = sample[2]
        # y = data_train.numpy();
        # print(y)
        # print(np.size(y))
        # np.save("tiff_zzz.npy", y)
        print(gt)
        print(gt.shape)








        
        # print("len_patch_lib:",len(patch_lib))
        # for i in sample:
        
            # print(i.shape)
            # A = i.squeeze(dim=0)
            # print(i.shape)
            # A = A.permute(1,2,0).numpy()
            # print(A.shape)
            # B= np.uint8(A)
            # cv2.imwrite('/ssd2/m3lab/usrs/crt/3D-ADS/imgs'+str(a)+'.png',B)
            # a = a+1
        k=0;