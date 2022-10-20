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


from self_sup_data.mvtec import SelfSupMVTecDataset, CLASS_NAMES, TEXTURES, OBJECTS
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
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'binary'}
    },
    'Shift-923874273' : {
        'fname': 'shift_923874273.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'binary'}
    },
    'Shift-2388222932' : {
        'fname': 'shift_2388222932.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'binary'}
    },
    'Shift-676346783' : {
        'fname': 'shift_676346783.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'binary'}
    },
    'Shift-123425' : {
        'fname': 'shift_123425.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'binary'}
    },
    'Shift-Intensity' : {
        'fname': 'shift_intensity.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-923874273' : {
        'fname': 'shift_intensity_923874273.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-2388222932' : {
        'fname': 'shift_intensity_2388222932.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-676346783' : {
        'fname': 'shift_intensity_676346783.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-123425' : {
        'fname': 'shift_intensity_123425.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Raw-Intensity' : {
        'fname': 'shift_raw_intensity.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'relu',
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-923874273' : {
        'fname': 'shift_raw_intensity_923874273.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-2388222932' : {
        'fname': 'shift_raw_intensity_2388222932.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-676346783' : {
        'fname': 'shift_raw_intensity_676346783.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-123425' : {
        'fname': 'shift_raw_intensity_123425.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'intensity'}
    },
    'Shift-M' : {
        'fname': 'shift_m.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'binary'}
    },
    'Shift-M-923874273' : {
        'fname': 'shift_m_923874273.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'binary'}
    },
    'Shift-M-2388222932' : {
        'fname': 'shift_m_2388222932.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'binary'}
    },
    'Shift-M-676346783' : {
        'fname': 'shift_m_676346783.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'binary'}
    },
    'Shift-M-123425' : {
        'fname': 'shift_m_123425.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'binary'}
    },
    'Shift-Intensity-M' : {
        'fname': 'shift_intensity_m.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-M-923874273' : {
        'fname': 'shift_intensity_m_923874273.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-M-2388222932' : {
        'fname': 'shift_intensity_m_2388222932.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-M-676346783' : {
        'fname': 'shift_intensity_m_676346783.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-M-123425' : {
        'fname': 'shift_intensity_m_123425.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Raw-Intensity-M' : {
        'fname': 'shift_raw_intensity_m.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'relu',
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-M-923874273' : {
        'fname': 'shift_raw_intensity_m_923874273.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-M-2388222932' : {
        'fname': 'shift_raw_intensity_m_2388222932.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-M-676346783' : {
        'fname': 'shift_raw_intensity_m_676346783.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-M-123425' : {
        'fname': 'shift_raw_intensity_m_123425.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'intensity'}
    },
### ------------------------------------ Foreign patch poisson blending / interpolation ------------------------------------ ###
'FPI-Poisson' : {
        'fname': 'fpi_poisson.pt',
        'out_dir': 'fpi/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':False, 
                           'shift':False, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'continuous'}
    },
'FPI' : {
        'fname': 'fpi.pt',
        'out_dir': 'fpi/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':False, 
                           'shift':False, 'same':False, 'mode':'uniform', 'label_mode':'continuous'}
    },
### ------------------------------------ Shifted patch pasting ------------------------------------ ###
'CutPaste' : {
        'fname': 'cut_paste.pt',
        'out_dir': 'cut_paste/',
        'loss': nn.BCELoss,
        'skip_background': True, 
        'final_activation': 'sigmoid',
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':False, 
                           'shift':True, 'same':True, 'mode':'swap', 'label_mode':'binary'}
    },
}

# note: these are half-widths in [0, 0.5]
# ((h_min, h_max), (w_min, w_max))
WIDTH_BOUNDS_PCT = {'bottle':((0.03, 0.4), (0.03, 0.4)), 'cable':((0.05, 0.4), (0.05, 0.4)), 'capsule':((0.03, 0.15), (0.03, 0.4)), 
                    'hazelnut':((0.03, 0.35), (0.03, 0.35)), 'metal_nut':((0.03, 0.4), (0.03, 0.4)), 'pill':((0.03, 0.2), (0.03, 0.4)), 
                    'screw':((0.03, 0.12), (0.03, 0.12)), 'toothbrush':((0.03, 0.4), (0.03, 0.2)), 'transistor':((0.03, 0.4), (0.03, 0.4)), 
                    'zipper':((0.03, 0.4), (0.03, 0.2)), 
                    'carpet':((0.03, 0.4), (0.03, 0.4)), 'grid':((0.03, 0.4), (0.03, 0.4)), 
                    'leather':((0.03, 0.4), (0.03, 0.4)), 'tile':((0.03, 0.4), (0.03, 0.4)), 'wood':((0.03, 0.4), (0.03, 0.4))}

MIN_OVERLAP_PCT = {'bottle': 0.25,  'capsule':0.25, 
                   'hazelnut':0.25, 'metal_nut':0.25, 'pill':0.25, 
                   'screw':0.25, 'toothbrush':0.25, 
                   'zipper':0.25}

MIN_OBJECT_PCT = {'bottle': 0.7,  'capsule':0.7, 
                  'hazelnut':0.7, 'metal_nut':0.5, 'pill':0.7, 
                  'screw':.5, 'toothbrush':0.25, 
                  'zipper':0.7}

NUM_PATCHES = {'bottle':3, 'cable':3, 'capsule':3, 'hazelnut':3, 'metal_nut':3, 
               'pill':3, 'screw':4, 'toothbrush':3, 'transistor':3, 'zipper':4,
               'carpet':4, 'grid':4, 'leather':4, 'tile':4, 'wood':4}

# k, x0 pairs
INTENSITY_LOGISTIC_PARAMS = {'bottle':(1/12, 24), 'cable':(1/12, 24), 'capsule':(1/2, 4), 'hazelnut':(1/12, 24), 'metal_nut':(1/3, 7), 
            'pill':(1/3, 7), 'screw':(1, 3), 'toothbrush':(1/6, 15), 'transistor':(1/6, 15), 'zipper':(1/6, 15),
            'carpet':(1/3, 7), 'grid':(1/3, 7), 'leather':(1/3, 7), 'tile':(1/3, 7), 'wood':(1/6, 15)}

# bottle is aligned but it's symmetric under rotation
UNALIGNED_OBJECTS = ['bottle', 'hazelnut', 'metal_nut', 'screw']

# non-aligned objects get extra time
EPOCHS = {'bottle':320, 'cable':320, 'capsule':320, 'hazelnut':560, 'metal_nut':560, 
          'pill':320, 'screw':560, 'toothbrush':320, 'transistor':320, 'zipper':320,
          'carpet':320, 'grid':320, 'leather':320, 'tile':320, 'wood':320}

# brightness, threshold pairs
BACKGROUND = {'bottle':(200, 60), 'screw':(200, 60), 'capsule':(200, 60), 'zipper':(200, 60), 
              'hazelnut':(20, 20), 'pill':(20, 20), 'toothbrush':(20, 20), 'metal_nut':(20, 20)}


def set_seed(seed_value):
    """Set seed for reproducibility.
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train(class_name, data_dir, out_dir, setting, device, pool, preact,
         min_lr = 1e-6, max_lr = 1e-3, batch_size = 64, seed = 1982342):
    set_seed(setting.get('seed', seed))
    num_epochs = EPOCHS.get(class_name)
    if setting.get('batch_size'):
        batch_size = setting.get('batch_size')
    # load data
    if class_name in UNALIGNED_OBJECTS:
        train_transform = T.Compose([
                T.RandomRotation(5),
                T.CenterCrop(230), 
                T.RandomCrop(224)])
        res = 256
    elif class_name in OBJECTS:
        # no rotation for aligned objects
        train_transform = T.Compose([
                T.CenterCrop(230), 
                T.RandomCrop(224)])
        res = 256
    else:  # texture
        train_transform = T.Compose([
                T.RandomVerticalFlip(), 
                T.RandomCrop(256)])
        res = 264
    train_dat = SelfSupMVTecDataset(root_path=data_dir, class_name=class_name, is_train=True, 
                                    low_res=res, download=False, transform=train_transform)

    train_dat.configure_self_sup(self_sup_args=setting.get('self_sup_args'))
    if setting.get('skip_background', False):
        train_dat.configure_self_sup(self_sup_args={'skip_background': BACKGROUND.get(class_name)})
    if class_name in TEXTURES:
        train_dat.configure_self_sup(self_sup_args={'resize_bounds': (.5, 2)})
    train_dat.configure_self_sup(on=True, self_sup_args={'width_bounds_pct': WIDTH_BOUNDS_PCT.get(class_name),
                                                         'intensity_logistic_params': INTENSITY_LOGISTIC_PARAMS.get(class_name),
                                                         'num_patches': NUM_PATCHES.get(class_name),
                                                         'min_object_pct': MIN_OBJECT_PCT.get(class_name),
                                                         'min_overlap_pct': MIN_OVERLAP_PCT.get(class_name)})
    
    loader_train = DataLoader(train_dat, batch_size, shuffle=True, num_workers=os.cpu_count(),
                              worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % 2**32))

    model = resnet18_enc_dec(num_classes=1, pool=pool, preact=preact, 
                             final_activation=setting.get('final_activation')).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=min_lr)
    loss_func = setting.get('loss')()

    out_dir = os.path.join(out_dir, setting.get('out_dir'), class_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_and_save_model(model, optimizer, loss_func, loader_train, class_name + '_'+ setting.get('fname'), out_dir, 
                         num_epochs=num_epochs, save_freq=80, device=device, scheduler=scheduler, save_intermediate_model=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", required=True, type=str)
    parser.add_argument("-d", "--data_dir", required=True, type=str)
    parser.add_argument("-s", "--setting", required=True, type=str)
    parser.add_argument("-c", "--categories", required=False, type=str, default='all')
    parser.add_argument("-n", "--class_name", required=False, type=str, default=None)
    parser.add_argument("--no_pool", required=False, action='store_true')
    parser.add_argument("--preact", required=False, action='store_true')
    args = parser.parse_args()

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_dir = args.data_dir

    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    setting = SETTINGS.get(args.setting)

    if args.class_name is not None:
        categories = [args.class_name]
    elif args.categories == 'texture':
        categories = TEXTURES
    elif args.categories == 'object':
        categories = OBJECTS
    else:
        categories = CLASS_NAMES

    for class_name in tqdm(categories):
        train(class_name, data_dir, out_dir, setting, device, not args.no_pool, args.preact)

