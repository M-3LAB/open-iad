import copy 
import random

__all__ = ['extract_transfer_data']

def extract_transfer_data(source_dataset, target_dataset, transfer_type='inter_class', 
                          target_train_num=2):

    source_sample_nums = [0] + source_dataset.sample_num_in_task
    source_sample_indice = source_dataset.sample_indices_in_task

    target_sample_nums = [0] + target_dataset.sample_num_in_task
    target_sample_indice = target_dataset.sample_indices_in_task

    if transfer_type == 'inter_class':
        pass
    elif transfer_type == 'intra_class':
        pass
    else:
        raise NotImplementedError('Transfer Type Has Not Been Implemented')

    # source normal training dataset 

    # target normal training dataset 

    # source anomaly training dataset 

    # target anomaly training dataset

    # obtain target number dataset  
    target_train_dataset = copy.deepcopy(target_dataset)
    for i, num in enumerate(target_dataset.sample_num_in_task): 
        if target_train_num > num:
            target_train_num = num 
        chosen_samples = random.sample()
        pass

