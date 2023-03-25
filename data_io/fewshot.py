import random
import copy
from torch.utils.data import Dataset


__all__ = ['FewShot', 'extract_fewshot_data']

class FewShot(Dataset):
    def __init__(self, data) -> None:
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def extract_fewshot_data(train_dataset, fewshot_exm=1):
    fewshot_exm_src = fewshot_exm
    # construct train_fewshot_dataset
    train_fewshot_dataset = copy.deepcopy(train_dataset)
    for i, num in enumerate(train_dataset.sample_num_in_task):
        if fewshot_exm > num:
            fewshot_exm = num
        chosen_samples = random.sample(train_fewshot_dataset.sample_indices_in_task[i], fewshot_exm)
        train_fewshot_dataset.sample_indices_in_task[i] = chosen_samples
        train_fewshot_dataset.sample_num_in_task[i] = fewshot_exm
        fewshot_exm = fewshot_exm_src

    return train_fewshot_dataset

