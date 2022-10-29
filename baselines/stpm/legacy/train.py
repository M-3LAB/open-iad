import argparse
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
import shutil
import time
from torchvision.models import resnet18
from PIL import Image
from sklearn.metrics import roc_auc_score

#imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

def data_transforms(load_size=256, mean_train=mean_train, std_train=std_train):
    data_transforms = transforms.Compose([
            transforms.Resize((load_size, load_size), Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.CenterCrop(input_size),
            transforms.Normalize(mean=mean_train,
                                std=std_train)])
    return data_transforms

def copy_files(src, dst, ignores=[]):
    src_files = os.listdir(src)
    for file_name in src_files:
        ignore_check = [True for i in ignores if i in file_name]
        if ignore_check:
            continue
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dst,file_name))
        if os.path.isdir(full_file_name):
            os.makedirs(os.path.join(dst, file_name), exist_ok=True)
            copy_files(full_file_name, os.path.join(dst, file_name), ignores)


def cal_loss(fs_list, ft_list, criterion):
    tot_loss = 0
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        _, _, h, w = fs.shape
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        # a_map = 1 - F.cosine_similarity(fs_norm, ft_norm)
        # f_loss = (1/(w*h))*torch.sum(a_map)
        f_loss = (0.5/(w*h))*criterion(fs_norm, ft_norm)
        tot_loss += f_loss

    return tot_loss

def cal_anomaly_map(fs_list, ft_list, out_size=224):
    anomaly_map = np.ones([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs_norm, ft_norm)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear')
        a_map = a_map[0,0,:,:].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        anomaly_map *= a_map
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    heatmap = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(heatmap) + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    

class STPM():
    def __init__(self):
        self.load_model()
        self.data_transform = data_transforms(load_size=load_size, mean_train=mean_train, std_train=std_train)

    def load_dataset(self):
        image_datasets = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'), transform=self.data_transform)
        self.dataloaders = DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=0) #, pin_memory=True)
        dataset_sizes = {'train': len(image_datasets)}    
        print('Dataset size : Train set - {}'.format(dataset_sizes['train']))    

    def load_model(self):
        self.features_t = []
        self.features_s = []
        def hook_t(module, input, output):
            self.features_t.append(output)
        def hook_s(module, input, output):
            self.features_s.append(output)
        
        self.model_t = resnet18(pretrained=True).to(device)
        self.model_t.layer1[-1].register_forward_hook(hook_t)
        self.model_t.layer2[-1].register_forward_hook(hook_t)
        self.model_t.layer3[-1].register_forward_hook(hook_t)

        self.model_s = resnet18(pretrained=False).to(device)
        self.model_s.layer1[-1].register_forward_hook(hook_s)
        self.model_s.layer2[-1].register_forward_hook(hook_s)
        self.model_s.layer3[-1].register_forward_hook(hook_s)
        
    def train(self):

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.model_s.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

        self.load_dataset()
        
        start_time = time.time()
        global_step = 0

        for epoch in range(num_epochs):
            print('-'*20)
            print('Time consumed : {}s'.format(time.time()-start_time))
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-'*20)
            self.model_t.eval()
            self.model_s.train()
            for idx, (batch, _) in enumerate(self.dataloaders): # batch loop
                global_step += 1
                batch = batch.to(device)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    self.features_t = []
                    self.features_s = []
                    _ = self.model_t(batch)
                    _ = self.model_s(batch)
                    # get loss using features.
                    loss = cal_loss(self.features_s, self.features_t, self.criterion)
                    loss.backward()
                    self.optimizer.step()

                if idx%2 == 0:
                    print('Epoch : {} | Loss : {:.4f}'.format(epoch, float(loss.data)))

        print('Total time consumed : {}'.format(time.time() - start_time))
        print('Train end.')
        if save_weight:
            print('Save weights.')
            torch.save(self.model_s.state_dict(), os.path.join(weight_save_path, 'model_s.pth'))

    def test(self):
        print('Test phase start')
        try:
            self.model_s.load_state_dict(torch.load(glob.glob(weight_save_path+'/*.pth')[0]))
        except:
            raise Exception('Check saved model path.')
        self.model_t.eval()
        self.model_s.eval()
        
        test_path = os.path.join(dataset_path, 'test')
        gt_path = os.path.join(dataset_path, 'ground_truth')
        test_imgs_all = glob.glob(test_path + '/**/*.png', recursive=True)
        test_imgs = [i for i in test_imgs_all if "good" not in i]
        test_imgs_good = [i for i in test_imgs_all if "good" in i]
        gt_imgs = glob.glob(gt_path + '/**/*.png', recursive=True)
        test_imgs.sort()
        gt_imgs.sort()
        gt_list_px_lvl = []
        gt_list_img_lvl = []
        pred_list_px_lvl = []
        pred_list_img_lvl = []
        start_time = time.time()
        print("Testset size : ", len(gt_imgs))        
        for i in range(len(test_imgs)):
            test_img_path = test_imgs[i]
            gt_img_path = gt_imgs[i]
            assert os.path.split(test_img_path)[1].split('.')[0] == os.path.split(gt_img_path)[1].split('_')[0], "Something wrong with test and ground truth pair!"
            defect_type = os.path.split(os.path.split(test_img_path)[0])[1]
            img_name = os.path.split(test_img_path)[1].split('.')[0]

            # ground truth
            gt_img_o = cv2.imread(gt_img_path,0)
            gt_img_o = cv2.resize(gt_img_o, (load_size, load_size))
            gt_img_o = gt_img_o[(load_size-input_size)//2:(load_size+input_size)//2,(load_size-input_size)//2:(load_size+input_size)//2]
            gt_list_px_lvl.extend(gt_img_o.ravel()//255)

            # load image
            test_img_o = cv2.imread(test_img_path)
            test_img_o = cv2.resize(test_img_o, (load_size, load_size))
            test_img_o = test_img_o[(load_size-input_size)//2:(load_size+input_size)//2,(load_size-input_size)//2:(load_size+input_size)//2]
            test_img = cv2.cvtColor(test_img_o, cv2.COLOR_BGR2RGB) # <~ here
            test_img = Image.fromarray(test_img)
            test_img = self.data_transform(test_img)
            test_img = torch.unsqueeze(test_img, 0).to(device)
            with torch.set_grad_enabled(False):
                self.features_t = []
                self.features_s = []
                _ = self.model_t(test_img)
                _ = self.model_s(test_img)
            # get anomaly map & each features
            anomaly_map, a_maps = cal_anomaly_map(self.features_s, self.features_t, out_size=input_size)
            pred_list_px_lvl.extend(anomaly_map.ravel())
            pred_list_img_lvl.append(np.max(anomaly_map))
            gt_list_img_lvl.append(1)

            # save anomaly map & features
            if args.save_anomaly_map:
                # normalize anomaly amp
                anomaly_map_norm = min_max_norm(anomaly_map)
                anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)
                # 64x64 map
                am64 = min_max_norm(a_maps[0])
                am64 = cvt2heatmap(am64*255)
                # 32x32 map
                am32 = min_max_norm(a_maps[1])
                am32 = cvt2heatmap(am32*255)
                # 16x16 map
                am16 = min_max_norm(a_maps[2])
                am16 = cvt2heatmap(am16*255)
                # anomaly map on image
                heatmap = cvt2heatmap(anomaly_map_norm*255)
                hm_on_img = heatmap_on_image(heatmap, test_img_o)

                # save images
                cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}.jpg'), test_img_o)
                cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_am64.jpg'), am64)
                cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_am32.jpg'), am32)
                cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_am16.jpg'), am16)
                cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_amap.jpg'), anomaly_map_norm_hm)
                cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_amap_on_img.jpg'), hm_on_img)
                cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_gt.jpg'), gt_img_o)
        
        # Test good image for image level score
        for i in range(len(test_imgs_good)):
            test_img_path = test_imgs_good[i]
            defect_type = os.path.split(os.path.split(test_img_path)[0])[1]
            img_name = os.path.split(test_img_path)[1].split('.')[0]

            # load image
            test_img_o = cv2.imread(test_img_path)
            test_img_o = cv2.resize(test_img_o, (load_size, load_size))
            test_img_o = test_img_o[(load_size-input_size)//2:(load_size+input_size)//2,(load_size-input_size)//2:(load_size+input_size)//2]
            test_img = cv2.cvtColor(test_img_o, cv2.COLOR_BGR2RGB) # <~ here
            test_img = Image.fromarray(test_img)
            test_img = self.data_transform(test_img)
            test_img = torch.unsqueeze(test_img, 0).to(device)
            with torch.set_grad_enabled(False):
                self.features_t = []
                self.features_s = []
                _ = self.model_t(test_img)
                _ = self.model_s(test_img)
            anomaly_map, a_maps = cal_anomaly_map(self.features_s, self.features_t, out_size=input_size)
            pred_list_img_lvl.append(np.max(anomaly_map))
            gt_list_img_lvl.append(0)
                    
        print('Total test time consumed : {}'.format(time.time() - start_time))
        print("Total pixel-level auc-roc score :")
        print(roc_auc_score(gt_list_px_lvl, pred_list_px_lvl))
        print("Total image-level auc-roc score :")
        print(roc_auc_score(gt_list_img_lvl, pred_list_img_lvl))

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', default='train')
    parser.add_argument('--dataset_path', default=r'/home/changwoo/hdd/datasets/mvtec_anomaly_detection/tile') #D:\Dataset\mvtec_anomaly_detection\transistor')
    parser.add_argument('--num_epoch', default=100)
    parser.add_argument('--lr', default=0.4)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--load_size', default=256)
    parser.add_argument('--input_size', default=256)
    parser.add_argument('--project_path', default=r'/home/changwoo/hdd/project_results/STPM_results') #D:\Project_Train_Results\mvtec_anomaly_detection\transistor_new_temp')
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    # print(torch.cuda.get_device_name(device))
    
    args = get_args()
    phase = args.phase
    dataset_path = args.dataset_path
    category = dataset_path.split('\\')[-1]
    num_epochs = args.num_epoch
    lr = args.lr
    batch_size = args.batch_size
    save_weight = True
    load_size = args.load_size
    input_size = args.input_size
    save_src_code = args.save_src_code
    project_path = args.project_path
    sample_path = os.path.join(project_path, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    weight_save_path = os.path.join(project_path, 'saved')
    if save_weight:
        os.makedirs(weight_save_path, exist_ok=True)
    if save_src_code:
        source_code_save_path = os.path.join(project_path, 'src')
        os.makedirs(source_code_save_path, exist_ok=True)
        copy_files('./', source_code_save_path, ['.git','.vscode','__pycache__','logs','README']) # copy source code
    

    stpm = STPM()
    if phase == 'train':
        stpm.train()
        stpm.test()
    elif phase == 'test':
        stpm.test()
    else:
        print('Phase argument must be train or test.')
