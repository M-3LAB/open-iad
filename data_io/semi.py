import copy
import random


__all__ = ['extract_semi_data']

def extract_semi_data(train_dataset, valid_dataset, anomaly_num=10, anomaly_overlap=False, upper_ratio=0.75):
    valid_sample_nums = [0] + valid_dataset.sample_num_in_task
    valid_sample_indice = valid_dataset.sample_indices_in_task

    # obtain noisy label in validation set
    noisy_indices = []
    for i in range(valid_dataset.num_task):
        anomaly_index = []
        for k, j in enumerate(range(valid_sample_nums[i], valid_sample_nums[i] + valid_sample_nums[i+1])):
            z = sum(valid_sample_nums[:i+1]) + k
            label = valid_dataset.labels_list[z]
            if label == 1:
                anomaly_index.append(valid_sample_indice[i][k])
        # set noisy data to be less than 50 percent of these in test set
        anomaly_num_max = int(len(anomaly_index) * upper_ratio)
        if anomaly_num >= anomaly_num_max:
            anomaly_num = anomaly_num_max
        noise_index = random.sample(anomaly_index, anomaly_num)

        noisy_indices.append(noise_index)

    # construct train_noisy_dataset
    valid_noisy_dataset = copy.deepcopy(valid_dataset)
    if not anomaly_overlap:
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