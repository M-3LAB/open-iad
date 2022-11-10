from abc import ABC
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torchvision.models import resnet18
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
import kornia.geometry.transform as kt
from tqdm import tqdm
import os

dataset_path = "/ssd-sata1/wjb/data/open-ad/mvtec2d"
categories = os.listdir(dataset_path)


class TrainDataset(Dataset, ABC):
    def __init__(self, path):
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        images_path = Path(path)
        images_list = list(images_path.glob('*.png'))  # list(images_path.glob('*.png'))
        images_list_str = [str(x) for x in images_list]
        self.images = images_list_str

    def __getitem__(self, item):
        image_path = self.images[item]
        image = Image.open(image_path)
        # If it is grayscale, it will be converted to RGB
        if image.mode == 'L':
            image = image.convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)


class TransformDataset(Dataset, ABC):
    def __init__(self, path, few_shot_num=1):
        self.few_shot_num = few_shot_num
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        images_path = Path(path)
        images_list = list(images_path.glob('*.png'))  # list(images_path.glob('*.png'))
        images_list_str = [str(x) for x in images_list]
        # Select few-shot images randomly
        self.images = np.random.choice(images_list_str, self.few_shot_num, replace=False)

    def __getitem__(self, item):
        image_path = self.images[item]
        image = Image.open(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')

        image = self.transform(image)
        return image

    def __len__(self):
        return self.few_shot_num


class TestDataset(Dataset, ABC):
    def __init__(self, is_good, path):
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if is_good:
            images_path = Path(path)
            images_list = list(images_path.glob('*.png'))
            images_list_str = [str(x) for x in images_list]
        else:
            bad_category = os.listdir(path)
            bad_category.remove('good')
            images_list_str = []
            for i in bad_category:
                images_path = Path(path + '/' + i)
                images_list = list(images_path.glob('*.png'))
                images_list_str += [str(x) for x in images_list]
        self.images = images_list_str

    def __getitem__(self, item):
        image_path = self.images[item]
        image = Image.open(image_path)
        # If it is grayscale, it will be converted to RGB
        if image.mode == 'L':
            image = image.convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)


class PatchCore:
    def __init__(self):
        self.features = None
        self.net = resnet18(pretrained=True).cuda()

    @staticmethod
    def embedding_concate(x, y):
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
        return z

    @staticmethod
    def reshape_embedding(embedding):
        embedding_list = []
        for k in range(embedding.shape[0]):
            for i in range(embedding.shape[2]):
                for j in range(embedding.shape[3]):
                    embedding_list.append(embedding[k, :, i, j])
        return embedding_list

    @staticmethod
    def feature_augmentation(features, angle=8):
        assert len(features) > 0, 'Feature Augmentation should be done in Original Features'
        angles_list = [45.0]
        if angle == 4:
            angles_list = [45.0, 135.0, 225.0]
        if angle == 8:
            angles_list = [45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        rot_feat_1 = features[0]
        rot_feat_2 = features[1]
        for angle in angles_list:
            angle = torch.tensor(angle).cuda()
            rot_feat_1 = torch.cat((rot_feat_1, kt.rotate(features[0], angle)), dim=0)
            rot_feat_2 = torch.cat((rot_feat_2, kt.rotate(features[1], angle)), dim=0)
        feature_rot = [rot_feat_1, rot_feat_2]
        return feature_rot

    def generate_patchcore_embedding(self, train_loader, is_few_shot, angle=2):
        self.net.eval()
        embeddings_list = []
        for i, data in enumerate(tqdm(train_loader, desc='Generating PatchCore Embedding')):
            img = data.cuda()
            embeddings = []
            # Get the feature from block2 and block3 of ResNet18
            for name, module in self.net.named_children():
                img = module(img)
                if name == 'layer2':
                    embeddings.append(img)
                if name == 'layer3':
                    embeddings.append(img)
                    break
            self.features = embeddings
            for j in range(len(embeddings)):
                pooling = torch.nn.AvgPool2d(3, 1, 1)
                embeddings[j] = pooling(embeddings[j])
            # Concatenate the feature of block2 and block3
            embedding = self.embedding_concate(embeddings[0], embeddings[1])
            embedding = embedding.cpu().detach().numpy()
            embedding = self.reshape_embedding(embedding)
            embeddings_list.extend(embedding)
            if is_few_shot:
                embeddings_rot = []
                embed_rot = self.feature_augmentation(self.features, angle)
                for j in range(len(embed_rot)):
                    pooling = torch.nn.AvgPool2d(3, 1, 1)
                    embeddings_rot.append(pooling(embed_rot[j]))
                embedding_rot = self.embedding_concate(embeddings_rot[0], embeddings_rot[1])
                embedding_rot = embedding_rot.cpu().detach().numpy()
                embedding_rot = self.reshape_embedding(embedding_rot)
                embeddings_list.extend(embedding_rot)
        total_embeddings = np.array(embeddings_list).astype(np.float32)
        # random_projection = SparseRandomProjection(n_components='auto', eps=0.9)
        # total_embeddings = random_projection.fit_transform(total_embeddings)
        return total_embeddings


if __name__ == '__main__':
    few_shot_num = 2
    augment_feature = True
    for category in categories:
        train_path = "/ssd-sata1/wjb/data/open-ad/mvtec2d/{}/train/good".format(category)
        test_good_path = "/ssd-sata1/wjb/data/open-ad/mvtec2d/{}/test/good".format(category)
        test_bad_path = "/ssd-sata1/wjb/data/open-ad/mvtec2d/{}/test".format(category)
        print('Processing category {}/{}: {}'.format(categories.index(category) + 1, len(categories), category))
        train_dataset = TrainDataset(train_path)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        train_embedding = PatchCore().generate_patchcore_embedding(train_loader, is_few_shot=False)
        print('Train Embedding Shape: ', train_embedding.shape)

        few_shot_dataset = TransformDataset(train_path, few_shot_num=few_shot_num)
        few_shot_loader = DataLoader(few_shot_dataset, batch_size=2, shuffle=True)
        few_shot_embedding_2 = PatchCore().generate_patchcore_embedding(few_shot_loader, is_few_shot=False, angle=2)
        print('Few Shot 2 Embedding Shape: ', few_shot_embedding_2.shape)

        few_shot_dataset = TransformDataset(train_path, few_shot_num=few_shot_num)
        few_shot_loader = DataLoader(few_shot_dataset, batch_size=2, shuffle=True)
        few_shot_embedding_4 = PatchCore().generate_patchcore_embedding(few_shot_loader, is_few_shot=augment_feature, angle=4)
        print('Few Shot 4 Embedding Shape: ', few_shot_embedding_4.shape)

        few_shot_dataset = TransformDataset(train_path, few_shot_num=few_shot_num)
        few_shot_loader = DataLoader(few_shot_dataset, batch_size=2, shuffle=True)
        few_shot_embedding_8 = PatchCore().generate_patchcore_embedding(few_shot_loader, is_few_shot=augment_feature, angle=8)
        print('Few Shot 8 Embedding Shape: ', few_shot_embedding_8.shape) 

        test_good_dataset = TestDataset(is_good=True, path=test_good_path)
        test_good_loader = DataLoader(test_good_dataset, batch_size=2, shuffle=True)
        test_good_embedding = PatchCore().generate_patchcore_embedding(test_good_loader, is_few_shot=False)
        print('Test Good Embedding Shape: ', test_good_embedding.shape)
        test_bad_dataset = TestDataset(is_good=False, path=test_bad_path)
        test_bad_loader = DataLoader(test_bad_dataset, batch_size=2, shuffle=True)
        test_bad_embedding = PatchCore().generate_patchcore_embedding(test_bad_loader, is_few_shot=False)
        print('Test Bad Embedding Shape: ', test_bad_embedding.shape)
        # Use TSNE to visualize the embedding
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        # Select 10% of the data to visualize randomly
        sample_ratio = 0.05
        train_embedding = train_embedding[
            np.random.choice(train_embedding.shape[0], int(train_embedding.shape[0] * sample_ratio), replace=False)]
        few_shot_embedding_2 = few_shot_embedding_2[np.random.choice(few_shot_embedding_2.shape[0], int(few_shot_embedding_2.shape[0] * sample_ratio), replace=False)]
        few_shot_embedding_4 = few_shot_embedding_4[np.random.choice(few_shot_embedding_4.shape[0], int(few_shot_embedding_4.shape[0] * sample_ratio), replace=False)]
        few_shot_embedding_8 = few_shot_embedding_8[np.random.choice(few_shot_embedding_8.shape[0], int(few_shot_embedding_8.shape[0] * sample_ratio), replace=False)]
        test_good_embedding = test_good_embedding[
            np.random.choice(test_good_embedding.shape[0], int(test_good_embedding.shape[0] * sample_ratio), replace=False)]
        test_bad_embedding = test_bad_embedding[
            np.random.choice(test_bad_embedding.shape[0], int(test_bad_embedding.shape[0] * sample_ratio), replace=False)]
        embeddings = np.concatenate((train_embedding, few_shot_embedding_2, few_shot_embedding_4, few_shot_embedding_8, test_good_embedding, test_bad_embedding), axis=0)
        print('Embedding Shape: ', embeddings.shape)
        # Use PCA to reduce the dimension of embedding to 50 (speed up the TSNE)
        print('PCA...')
        pca = PCA(n_components=50)
        embeddings = pca.fit_transform(embeddings)
        print('Start TSNE')
        embeddings = tsne.fit_transform(embeddings)
        print('TSNE Done')

        a = train_embedding.shape[0] 
        b = few_shot_embedding_2.shape[0] 
        c = few_shot_embedding_4.shape[0] 
        d = few_shot_embedding_8.shape[0] 
        e = test_good_embedding.shape[0] 
        f = test_bad_embedding.shape[0] 

        train_embedding = embeddings[:a]
        few_shot_embedding_2 = embeddings[a: a+b]
        few_shot_embedding_4 = embeddings[a+b: a+b+c]
        few_shot_embedding_8 = embeddings[a+b+c: a+b+c+d]
        test_good_embedding = embeddings[a+b+c+d: a+b+c+d+f]
        test_bad_embedding = embeddings[a+b+c+d+f:]

        fig = plt.figure(figsize=(5, 4))
        plt.scatter(train_embedding[:, 0], train_embedding[:, 1], c='y', label='Train', s=2, alpha=0.4)
        plt.scatter(test_good_embedding[:, 0], test_good_embedding[:, 1], c='g', label='Test-Good', s=2, alpha=0.4)
        plt.scatter(test_bad_embedding[:, 0], test_bad_embedding[:, 1], c='r', label='Test-Bad', s=2, alpha=1)
        plt.scatter(few_shot_embedding_2[:, 0], few_shot_embedding_2[:, 1], c='b', label='Few-Shot', s=4, alpha=1)
        plt.legend(fontsize='medium')
        plt.xticks([])
        plt.yticks([])
        plt.title('{} (only 2 Shot)'.format(category), fontsize=8)
        plt.savefig('./work_dir/vis_result/{}_{}_not_augment.png'.format(category, 2), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
    
    
        fig = plt.figure(figsize=(5, 4))
        plt.scatter(train_embedding[:, 0], train_embedding[:, 1], c='y', label='Train', s=2, alpha=0.4)
        # plt.scatter(test_good_embedding[:, 0], test_good_embedding[:, 1], c='g', label='Test-Good', s=2, alpha=1)
        # plt.scatter(test_bad_embedding[:, 0], test_bad_embedding[:, 1], c='r', label='Test-Bad', s=2, alpha=1)
        plt.scatter(few_shot_embedding_4[:, 0], few_shot_embedding_4[:, 1], c='b', label='Few-Shot', s=4, alpha=1)
        plt.legend(fontsize='medium')
        plt.xticks([])
        plt.yticks([])
        plt.title('{} (2 Shot with 4 Angles)'.format(category), fontsize=8)
        plt.savefig('./work_dir/vis_result/{}_{}_augment.png'.format(category, 4), bbox_inches='tight', pad_inches=0.1)
        plt.clf()

        fig = plt.figure(figsize=(5, 4))
        plt.scatter(train_embedding[:, 0], train_embedding[:, 1], c='y', label='Train', s=2, alpha=0.4)
        # plt.scatter(test_good_embedding[:, 0], test_good_embedding[:, 1], c='g', label='Test-Good', s=2, alpha=1)
        # plt.scatter(test_bad_embedding[:, 0], test_bad_embedding[:, 1], c='r', label='Test-Bad', s=2, alpha=1)
        plt.scatter(few_shot_embedding_8[:, 0], few_shot_embedding_8[:, 1], c='b', label='Few-Shot', s=4, alpha=1)
        plt.legend(fontsize='medium')
        plt.xticks([])
        plt.yticks([])
        plt.title('{} (2 Shot with 8 Angles)'.format(category), fontsize=8)
        plt.savefig('./work_dir/vis_result/{}_{}_augment.png'.format(category, 8), bbox_inches='tight', pad_inches=0.1)
        plt.clf()