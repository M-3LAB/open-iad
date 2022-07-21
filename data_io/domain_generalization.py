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

def domain_gen(config, data):
    fewshot_dg_datset = FewShot(data)
    train_fewshot_loader = DataLoader(fewshot_dg_datset,
                            batch_size=config['batch_size'],
                            num_workers=config['num_workers'])
    device = config['gpu_id']

    if config['backbone'] == 'resnet18':
        backbone = models.resnet18(pretrained=True, progress=True).to(device)
    elif config['backbone'] == 'wide_resnet50':
        backbone = models.wide_resnet50_2(pretrained=True, progress=True).to(device)
    else:
        raise NotImplementedError('This Pretrained Model Not Implemented Error')

    features = []

    def get_layer_features():

        def hook_t(module, input, output):
            features.append(output)
        
        #self.backbone.layer1[-1].register_forward_hook(hook_t)
        backbone.layer2[-1].register_forward_hook(hook_t)
        backbone.layer3[-1].register_forward_hook(hook_t)

    get_layer_features()

    with torch.no_grad():
        for batch_id, batch in enumerate(train_fewshot_loader):
            img = batch['img'].to(device)
            mask = batch['mask'].to(device)
            label = batch['label'].to(device)
            # extract features from backbone
            features.clear()
            _ = backbone(img)

            # pooling for layer 2 and layer 3 features
            embeddings = []
            for feat in features:
                pooling = torch.nn.AvgPool2d(3, 1, 1)
                embeddings.append(pooling(feat))

            embedding = PatchCore2D.embedding_concate(embeddings[0], embeddings[1])
            embedding = PatchCore2D.reshape_embedding(embedding.detach().numpy())
            embedding = np.array(embedding)
        # img, label, mask, task_id = self.data[idx]['img'], self.data[idx]['label'], self.data[idx]['mask'], self.data[idx]['task_id']
        # return {
        #     'img': img, 'label':label, 'mask':mask, 'task_id':task_id
        # }
    return data
    