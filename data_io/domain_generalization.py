import torch
import os
import random
from PIL import Image
import numpy as np
from torchvision import models
from torchvision import transforms as T
from data_io.mvtec2d import FewShot
from torch.utils.data import DataLoader
from arch_base.patchcore2d import PatchCore2D
import cv2
import torchvision



def domain_gen(config, data):

    # t = {'degrees': [90, 180], 'translate': [0, 0], 'scale': [1.0, 1.0]}
    # transform = T.Compose([T.ToPILImage(),
    #                         T.RandomAffine(degrees=t['degrees'], translate=t['translate'], scale=t['scale'], fillcolor=0),
    #                         T.ToTensor(),
    #                         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # mask_transform = T.Compose([T.ToPILImage(),
    #                         T.RandomAffine(degrees=t['degrees'], translate=t['translate'], scale=t['scale'], fillcolor=0),
    #                         T.ToTensor()])
    degrees = [(0, 0), (90, 90), (180, 180), (270, 270)]
    data_dg = []
    for d in data:
        data_dg.append(d)
        img = PatchCore2D.torch_to_cv(d['img'].unsqueeze(0)).astype(np.uint8)
        mask_img = d['mask']
        for degree in degrees:
            img_da = T.ToPILImage()(img)
            img_da = T.RandomAffine(degrees=degree, fillcolor=(0, 0, 0))(img_da)
            img_da = T.PILToTensor()(img_da)
            # visualize
            img_da_cv = img_da.numpy().transpose(1, 2, 0)
            cv2.imwrite('./legacy_code/img_{}.png'.format(degree[0]), img_da_cv)
    
            cv_img = cv2.cvtColor(img_da_cv, cv2.COLOR_RGB2BGR)
            img_da = torch.from_numpy(cv_img).permute(2, 0, 1)
            img_da = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_da.to(torch.float32))

            mask = T.ToPILImage()(mask_img)
            mask = T.RandomAffine(degrees=degree, fillcolor= 0)(mask)
            mask = T.PILToTensor()(mask)
            data_dg.append({'img': img_da, 'label': d['label'], 'mask': mask, 'task_id': d['task_id']})
    data = data_dg

    # fewshot_dg_datset = FewShot(data)
    # train_fewshot_loader = DataLoader(fewshot_dg_datset,
    #                         batch_size=config['batch_size'],
    #                         num_workers=config['num_workers'])
    # device = config['gpu_id']

    # if config['backbone'] == 'resnet18':
    #     backbone = models.resnet18(pretrained=True, progress=True).to(device)
    # elif config['backbone'] == 'wide_resnet50':
    #     backbone = models.wide_resnet50_2(pretrained=True, progress=True).to(device)
    # else:
    #     raise NotImplementedError('This Pretrained Model Not Implemented Error')

    # features = []

    # def get_layer_features():

    #     def hook_t(module, input, output):
    #         features.append(output)
        
    #     #self.backbone.layer1[-1].register_forward_hook(hook_t)
    #     backbone.layer2[-1].register_forward_hook(hook_t)
    #     backbone.layer3[-1].register_forward_hook(hook_t)

    # get_layer_features()

    # fewshot_embedding = None
    # with torch.no_grad():
    #     for batch_id, batch in enumerate(train_fewshot_loader):
    #         img = batch['img'].to(device)
    #         mask = batch['mask'].to(device)
    #         label = batch['label'].to(device)
    #         # extract features from backbone
    #         features.clear()
    #         _ = backbone(img)

    #         # pooling for layer 2 and layer 3 features
    #         embeddings = []
    #         for feat in features:
    #             pooling = torch.nn.AvgPool2d(3, 1, 1)
    #             embeddings.append(pooling(feat))

    #         embedding = PatchCore2D.embedding_concate(embeddings[0], embeddings[1])
    #         embedding = PatchCore2D.reshape_embedding(embedding.detach().numpy())
    #         embedding = np.array(embedding)

    # fewshot_embedding = embedding.reshape(-1, 784, 1536).swapaxes(0, 1)
    # mean_x = np.mean(fewshot_embedding, 1)
    # std_x = np.std(fewshot_embedding, 1)

    # data = []
    # for i in range(1000):
    #     sampled_x = []
    #     for mu, sigma in zip(mean_x.flatten(), std_x.flatten()):
    #         s = np.random.normal(mu, sigma)
    #         sampled_x.append(s)

    #     sampled_x = np.array(sampled_x).reshape(784, 1536)
    #     data.append(sampled_x)

    return data
    