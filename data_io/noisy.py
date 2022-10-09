from torch.utils.data import Dataset
from torchvision import transforms as T


from data_io.mvtec2d import MVTec2D
from data_io.mpdd import MPDD
from data_io.mvteclogical import MVTecLogical


__all__ = ['MVTec2DNoisy', 'MPDDNoisy', 'MVTecLogicalNoisy']

class MVTec2DNoisy(MVTec2D):
    def __init__(self, noisy_ratio=0.1, noisy_overlap=False, ):
        super().__init__()
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

class MPDDNoisy(MPDD):
    def __init__(self, noisy_ratio=0.1, noisy_overlap=False, ):
        super().__init__()
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class MVTecLogicalNoisy(MVTecLogical):
    def __init__(self, noisy_ratio=0.1, noisy_overlap=False, ):
        super().__init__()
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
