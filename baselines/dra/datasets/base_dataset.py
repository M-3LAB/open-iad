from torch.utils.data import Dataset

class BaseADDataset(Dataset):

    def __init__(self):
        super(BaseADDataset).__init__()

        self.normal_idx = None
        self.outlier_idx = None