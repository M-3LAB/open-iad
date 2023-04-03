import copy 
import random

__all__ = ['extract_transfer_data']

def extract_transfer_data(source_domain_dataset, target_domain_dataset, transfer_type):
    source_sample_nums = [0] + source_domain_dataset.sample_num_in_task
    source_sample_indice = source_domain_dataset.sample_indices_in_task

    target_sample_nums = [0] + target_domain_dataset.sample_num_in_task
    target_sample_indice = target_domain_dataset.sample_indices_in_task

    if transfer_type == 'inter_class':
        pass
    elif transfer_type == 'intra_class':
        pass
    else:
        raise NotImplementedError('Transfer Type Has Not Been Implemented')

    # obtain target number dataset  

