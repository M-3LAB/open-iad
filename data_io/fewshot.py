from pyclbr import readmodule
import random
import copy
from torch.utils.data import Dataset

from data_io.mvtec2d import MVTec2D
from data_io.mpdd import MPDD
from data_io.mvtec2df3d import MVTec2DF3D
from data_io.mvtecloco import MVTecLoco
from data_io.mtd import MTD
from data_io.btad import BTAD


__all__ = ['FewShot', 'extract_fewshot_data', 'MVTec2DFewShot', 'MPDDFewShot', 'MVTecLocoFewShot',
           'MTDFewShot', 'BTADFewShot', 'MVTec2DF3DFewShot']

class FewShot(Dataset):
    def __init__(self, data) -> None:
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def extract_fewshot_data(train_dataset, fewshot_exm=1):
    # construct train_fewshot_dataset
    train_fewshot_dataset = copy.deepcopy(train_dataset)
    for i, num in enumerate(train_dataset.sample_num_in_task):
        if fewshot_exm > num:
            fewshot_exm = num
        chosen_samples = random.sample(train_fewshot_dataset.sample_indices_in_task[i], fewshot_exm)
        train_fewshot_dataset.sample_indices_in_task[i] = chosen_samples
        train_fewshot_dataset.sample_num_in_task[i] = fewshot_exm

    return train_fewshot_dataset


class MVTec2DFewShot(MVTec2D):
    def __init__(self, data_path, learning_mode='centralized', phase='train', 
                 data_transform=None, num_task=15, fewshot_exm=1):
        self.fewshot_exm = fewshot_exm
        super(MVTec2DFewShot, self).__init__(data_path=data_path, learning_mode=learning_mode, phase=phase, 
                 data_transform=data_transform, num_task=num_task)
    
    def allocate_task_data(self):
        start = 0
        for num in self.sample_num_in_task:
            end = start + num
            indice = [i for i in range(start, end)]
            random.shuffle(indice)
            self.sample_indices_in_task.append(indice[:self.fewshot_exm])
            start = end


class MPDDFewShot(MPDD):
    def __init__(self, data_path, learning_mode='centralized', phase='train', 
                 data_transform=None, num_task=6, fewshot_exm=1):
        self.fewshot_exm = fewshot_exm
        super(MPDDFewShot, self).__init__(data_path=data_path, learning_mode=learning_mode, phase=phase, 
                 data_transform=data_transform, num_task=num_task)
    
    def allocate_task_data(self):
        start = 0
        for num in self.sample_num_in_task:
            end = start + num
            indice = [i for i in range(start, end)]
            random.shuffle(indice)
            self.sample_indices_in_task.append(indice[:self.fewshot_exm])
            start = end

class MVTecLocoFewShot(MVTecLoco):
    def __init__(self, data_path, ignore_anomaly_type='logical_anomalies',learning_mode='centralized', phase='train', 
                 data_transform=None, num_task=5, fewshot_exm=16):
        self.fewshot_exm = fewshot_exm
        super(MVTecLocoFewShot, self).__init__(data_path=data_path, ignore_anomaly_type=ignore_anomaly_type, learning_mode=learning_mode, phase=phase, 
                 data_transform=data_transform, num_task=num_task)
    
    def allocate_task_data(self):
        start = 0
        for num in self.sample_num_in_task:
            end = start + num
            indice = [i for i in range(start, end)]
            random.shuffle(indice)
            self.sample_indices_in_task.append(indice[:self.fewshot_exm])
            start = end

class MTDFewShot(MTD):
    def __init__(self, data_path, learning_mode='centralized', phase='train', 
                 data_transform=None, num_task=1, fewshot_exm=1):
        self.fewshot_exm = fewshot_exm
        super(MTDFewShot, self).__init__(data_path=data_path, learning_mode=learning_mode, phase=phase, 
                 data_transform=data_transform, num_task=num_task)
    
    def allocate_task_data(self):
        start = 0
        for num in self.sample_num_in_task:
            end = start + num
            indice = [i for i in range(start, end)]
            random.shuffle(indice)
            self.sample_indices_in_task.append(indice[:self.fewshot_exm])
            start = end

class BTADFewShot(BTAD):
    def __init__(self, data_path, learning_mode='centralized', phase='train', 
                 data_transform=None, num_task=1, fewshot_exm=1):
        self.fewshot_exm = fewshot_exm
        super(BTADFewShot, self).__init__(data_path=data_path, learning_mode=learning_mode, phase=phase, 
                 data_transform=data_transform, num_task=num_task)
    
    def allocate_task_data(self):
        start = 0
        for num in self.sample_num_in_task:
            end = start + num
            indice = [i for i in range(start, end)]
            random.shuffle(indice)
            self.sample_indices_in_task.append(indice[:self.fewshot_exm])
            start = end

class MVTec2DF3DFewShot(MVTec2DF3D):
    def __init__(self, data_path, learning_mode='centralized', phase='train', 
                 data_transform=None, num_task=15, fewshot_exm=1):
        self.fewshot_exm = fewshot_exm
        super(MVTec2DF3DFewShot, self).__init__(data_path=data_path, learning_mode=learning_mode, phase=phase, 
                 data_transform=data_transform, num_task=num_task)
    
    def allocate_task_data(self):
        start = 0
        for num in self.sample_num_in_task:
            end = start + num
            indice = [i for i in range(start, end)]
            random.shuffle(indice)
            self.sample_indices_in_task.append(indice[:self.fewshot_exm])
            start = end