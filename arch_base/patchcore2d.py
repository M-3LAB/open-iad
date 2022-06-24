import torch
import torch.nn as nn
from models.patchcore.patchcore import PatchCore

__all__ = ['PatchCore2D']

class PatchCore2D():
    def __init__(self, config, train_loaders, valid_loaders, device):
        
        self.config = config
        self.train_loaders = train_loaders
        self.valid_loaders = valid_loaders
        self.device = device

        #Model 
        self.model = PatchCore(input_size=self.config['input_size'],
                               backbone=self.config['backbone'],
                               layers=self.config['layers'],
                               num_neighbours=self.config['num_neighbours']).to(self.device)
        
        self.coreset_sampling_ratio = self.config['sampling_ratio'] 
        self.embeddings = []
        
    def train_epoch(self, inf=''):
        self.model.train()
        # Extract features for each image 
        self.model.feature_extractor.eval()

        for epoch in self.config['num_epoch']:
            for task_idx, train_loader in enumerate(self.train_loaders):
                print('run task: {}'.format(task_idx))
                for batch_id, batch in enumerate(train_loader):
                    if self.config['debug'] and batch_id > self.batch_limit:
                        break
                    img = batch['image'].to(self.device)

                    embedding = self.model(img)
                    self.embeddings.append(embedding)                    
                    
                    
    def prediction(self):
        self.model.eval()

        print(f'Aggregating the embedding extracted from the training set')
        embeddings_list = torch.vstack(self.embeddings)
        
        print(f'Applying coreset subsampling to get the embedding')
        #Obtain the memory bank
        self.model.subsample_embedding(embeddings_list, self.coreset_sampling_ratio)

        with torch.no_grad():
            for task_idx, valid_loader in enumerate(self.valid_loaders):
                print(f'run task: {task_idx}')
                for batch_id, batch in enumerate(valid_loader):
                    if self.config['debug'] and batch_id > self.batch_limit:
                        break

                    img = batch['image'].to(self.device)

                    anomaly_maps, anomaly_score = self.model(img)
                    pred_scores = anomaly_score.unsqueeze(0)

