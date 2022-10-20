import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, sampler
import matplotlib.pyplot as plt
from torchvision import transforms as T
import argparse
from tqdm import tqdm
import cv2


from self_sup_data.chest_xray import SelfSupChestXRay
from model.resnet import resnet18_enc_dec
from experiments.training_utils import train_and_save_model


SETTINGS = {
### ------------------------------------------------ NSA ------------------------------------------------ ###
    'Shift' : {
        'fname': 'shift.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'binary'}
    },
    'Shift-923874273' : {
        'fname': 'shift_923874273.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'binary'}
    },
    'Shift-2388222932' : {
        'fname': 'shift_2388222932.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'binary'}
    },
    'Shift-676346783' : {
        'fname': 'shift_676346783.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'binary'}
    },
    'Shift-123425' : {
        'fname': 'shift_123425.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'binary'}
    },
    'Shift-Intensity' : {
        'fname': 'shift_intensity.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-923874273' : {
        'fname': 'shift_intensity_923874273.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-2388222932' : {
        'fname': 'shift_intensity_2388222932.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-676346783' : {
        'fname': 'shift_intensity_676346783.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-123425' : {
        'fname': 'shift_intensity_123425.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Raw-Intensity' : {
        'fname': 'shift_raw_intensity.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'relu',
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-923874273' : {
        'fname': 'shift_raw_intensity_923874273.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-2388222932' : {
        'fname': 'shift_raw_intensity_2388222932.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-676346783' : {
        'fname': 'shift_raw_intensity_676346783.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-123425' : {
        'fname': 'shift_raw_intensity_123425.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'intensity'}
    },
    'Shift-M' : {
        'fname': 'shift_m.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'binary'}
    },
    'Shift-M-923874273' : {
        'fname': 'shift_m_923874273.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'binary'}
    },
    'Shift-M-2388222932' : {
        'fname': 'shift_m_2388222932.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'binary'}
    },
    'Shift-M-676346783' : {
        'fname': 'shift_m_676346783.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'binary'}
    },
    'Shift-M-123425' : {
        'fname': 'shift_m_123425.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'binary'}
    },
    'Shift-Intensity-M' : {
        'fname': 'shift_intensity_m.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-M-923874273' : {
        'fname': 'shift_intensity_m_923874273.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-M-2388222932' : {
        'fname': 'shift_intensity_m_2388222932.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-M-676346783' : {
        'fname': 'shift_intensity_m_676346783.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-M-123425' : {
        'fname': 'shift_intensity_m_123425.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Raw-Intensity-M' : {
        'fname': 'shift_raw_intensity_m.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'relu',
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-M-923874273' : {
        'fname': 'shift_raw_intensity_m_923874273.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-M-2388222932' : {
        'fname': 'shift_raw_intensity_m_2388222932.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-M-676346783' : {
        'fname': 'shift_raw_intensity_m_676346783.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-M-123425' : {
        'fname': 'shift_raw_intensity_m_123425.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'resize':True, 'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'intensity'}
    },
### ------------------------------------ Foreign patch poisson blending / interpolation ------------------------------------ ###
    'FPI-Poisson' : {
        'fname': 'fpi_poisson.pt',
        'out_dir': 'fpi/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'self_sup_args' : { 'resize':False, 'shift':False, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'continuous'}
    },
    'FPI-Poisson-923874273' : {
        'fname': 'fpi_poisson_923874273.pt',
        'out_dir': 'fpi/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : { 'resize':False, 'shift':False, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'continuous'}
    },
    'FPI-Poisson-2388222932' : {
        'fname': 'fpi_poisson_2388222932.pt',
        'out_dir': 'fpi/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : { 'resize':False, 'shift':False, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'continuous'}
    },
    'FPI-Poisson-676346783' : {
        'fname': 'fpi_poisson_676346783.pt',
        'out_dir': 'fpi/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : { 'resize':False, 'shift':False, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'continuous'}
    },
    'FPI-Poisson-123425' : {
        'fname': 'fpi_poisson_123425.pt',
        'out_dir': 'fpi/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : { 'resize':False, 'shift':False, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'continuous'}
    },
    'FPI' : {
        'fname': 'fpi.pt',
        'out_dir': 'fpi/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'self_sup_args' : {'resize':False, 'shift':False, 'same':False, 'mode':'uniform', 'label_mode':'continuous'}
    },
### ------------------------------------ Shifted patch pasting ------------------------------------ ###
    'CutPaste' : {
        'fname': 'cut_paste.pt',
        'out_dir': 'cut_paste/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'self_sup_args' : {'resize':False, 'shift':True, 'same':True, 'mode':'swap', 'label_mode':'binary'}
    },
}

# ((h_min, h_max), (w_min, w_max))
# note: this is half-width not width
WIDTH_BOUNDS_PCT = ((0.03, 0.4), (0.03, 0.4))

GAMMA_PARAMS = (2, 0.03, 0.05)

MIN_OVERLAP_PCT = 0.7

MIN_OBJECT_PCT = 0.7

NUM_PATCHES = 3

# k, x0 
INTENSITY_LOGISTIC_PARAMS = (1/2, 4)

# brightness, threshold pairs
BACKGROUND = (0, 20)


def set_seed(seed_value):
    """Set seed for reproducibility.
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train(data_dir, file_list, out_dir, setting, device, pool, preact,
          min_lr = 1e-6, max_lr = 1e-3, batch_size = 64, seed = 1982342, num_epochs=240):
    set_seed(setting.get('seed', seed))
    train_transform = T.Compose([
            T.RandomRotation(3),
            T.CenterCrop(230), 
            T.RandomCrop(224)])
    train_dat = SelfSupChestXRay(data_dir=data_dir, normal_files=file_list, is_train=True, res=256, transform=train_transform)

    train_dat.configure_self_sup(self_sup_args=setting.get('self_sup_args'))
    train_dat.configure_self_sup(on=True, self_sup_args={'width_bounds_pct': WIDTH_BOUNDS_PCT,
                                                         'intensity_logistic_params': INTENSITY_LOGISTIC_PARAMS,
                                                         'num_patches': NUM_PATCHES,
                                                         'skip_background': BACKGROUND,
                                                         'min_object_pct': MIN_OBJECT_PCT,
                                                         'min_overlap_pct': MIN_OVERLAP_PCT,
                                                         'gamma_params': GAMMA_PARAMS,
                                                         'verbose': False})
    
    loader_train = DataLoader(train_dat, batch_size, shuffle=True, num_workers=os.cpu_count(),
                              worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % 2**32))

    model = resnet18_enc_dec(num_classes=1, pool=pool, preact=preact, in_channels=1,
                             final_activation=setting.get('final_activation')).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=min_lr)
    loss_func = setting.get('loss')()

    out_dir = os.path.join(out_dir, setting.get('out_dir'))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_and_save_model(model, optimizer, loss_func, loader_train, setting.get('fname'), out_dir, 
                         num_epochs=num_epochs, save_freq=5, device=device, scheduler=scheduler, save_intermediate_model=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", required=True, type=str)
    parser.add_argument("-l", "--file_list", required=True, type=str)
    parser.add_argument("-o", "--out_dir", required=True, type=str)
    parser.add_argument("-s", "--setting", required=True, type=str)
    parser.add_argument("--no_pool", required=False, action='store_true')
    parser.add_argument("--preact", required=False, action='store_true')
    args = parser.parse_args()

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    setting = SETTINGS.get(args.setting)

    with open(args.file_list, "r") as f:
        flist = f.read().splitlines()
    train(args.data_dir, flist, out_dir, setting, device, not args.no_pool, args.preact)
