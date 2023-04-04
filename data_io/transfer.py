import copy 
import random

__all__ = ['extract_transfer_data']

def extract_transfer_data(
                          source_train_dataset, 
                          source_valid_dataset, 
                          target_train_dataset, 
                          target_valid_dataset,
                          transfer_type='inter_class', 
                          target_train_num=2
                          ):

    source_train_sample_nums = [0] + source_train_dataset.sample_num_in_task
    source_train_sample_indice = source_train_dataset.sample_indices_in_task

    source_valid_sample_nums = [0] + source_valid_dataset.sample_num_in_task
    source_valid_sample_indice = source_valid_dataset.sample_indices_in_task

    target_train_sample_nums = [0] + target_train_dataset.sample_num_in_task
    target_train_sample_indice = target_train_dataset.sample_indices_in_task

    target_valid_sample_nums = [0] + target_valid_dataset.sample_num_in_task
    target_valid_sample_indice = target_valid_dataset.sample_indices_in_task

    if transfer_type == 'inter_class':
        pass
    elif transfer_type == 'intra_class':
        pass
    else:
        raise NotImplementedError('Transfer Type Has Not Been Implemented')

    # construct source normal training dataset 
    

    # construct target normal training dataset 

    # construct source anomaly training dataset 

    # construct target anomaly training dataset

    # obtain target number dataset  

    #target_train_dataset = copy.deepcopy(target_dataset)
    #for i, num in enumerate(target_dataset.sample_num_in_task): 
    #    if target_train_num > num:
    #        target_train_num = num 
    #    chosen_samples = random.sample()
    #    pass

