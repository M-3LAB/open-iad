import copy
import random


__all__ = ['extract_noisy_data']

def extract_noisy_data(train_dataset, valid_dataset, noisy_ratio=0.1, noisy_overlap=False, upper_ratio=0.75):
    valid_sample_nums = [0] + valid_dataset.sample_num_in_task
    valid_sample_indice = valid_dataset.sample_indices_in_task
    
    # obtain noisy label in validation set
    noisy_indices = []
    for task_id in range(valid_dataset.num_task):
        anomaly_index = []
        for k, j in enumerate(range(valid_sample_nums[task_id], valid_sample_nums[task_id] + valid_sample_nums[task_id+1])):
            z = sum(valid_sample_nums[:task_id+1]) + k
            label = valid_dataset.labels_list[z]
            if label == 1:
                anomaly_index.append(valid_sample_indice[task_id][k])
        # set noisy data to be less than 75 percent of these in test set
        noise_num = int(noisy_ratio * train_dataset.sample_num_in_task[task_id])
        anomaly_num_max = int(len(anomaly_index) * upper_ratio)
        if noise_num >= anomaly_num_max:
            noise_num = anomaly_num_max
        noise_index = random.sample(anomaly_index, noise_num)

        noisy_indices.append(noise_index)

    # construct valid_noisy_dataset 
    valid_noisy_dataset = copy.deepcopy(valid_dataset)
    if not noisy_overlap:
        for task_id in range(valid_dataset.num_task):
            valid_noisy_dataset.sample_indices_in_task[task_id] = list(set(valid_sample_indice[task_id]) - set(noisy_indices[task_id]))
            valid_noisy_dataset.sample_num_in_task[task_id] = len(valid_noisy_dataset.sample_indices_in_task[task_id])

    # construct noisy_dataset 
    noisy_dataset = copy.deepcopy(valid_dataset)
    for task_id in range(valid_dataset.num_task):
        noisy_dataset.sample_indices_in_task[task_id] = noisy_indices[task_id] 
        noisy_dataset.sample_num_in_task[task_id] = len(noisy_dataset.sample_indices_in_task[task_id])

    # construct train_noisy_dataset
    train_noisy_dataset = copy.deepcopy(train_dataset)
    for task_id in range(valid_dataset.num_task):
        for img_id in noisy_indices[task_id]:
            train_noisy_dataset.imgs_list.append(noisy_dataset.imgs_list[img_id]) 
            train_noisy_dataset.labels_list.append(noisy_dataset.labels_list[img_id]) 
            train_noisy_dataset.masks_list.append(noisy_dataset.masks_list[img_id]) 
            train_noisy_dataset.task_ids_list.append(noisy_dataset.task_ids_list[img_id]) 

    for task_id in range(train_dataset.num_task):
        noisy_indices = []
        for i in range(noisy_dataset.sample_num_in_task[task_id]):
            local_idx = i + len(train_dataset.imgs_list) + sum(noisy_dataset.sample_num_in_task[:task_id])
            noisy_indices.append(int(local_idx))

        train_noisy_dataset.sample_indices_in_task[task_id].extend(noisy_indices)
        train_noisy_dataset.sample_num_in_task[task_id] += noisy_dataset.sample_num_in_task[task_id] 

    return train_noisy_dataset, valid_noisy_dataset, noisy_dataset
