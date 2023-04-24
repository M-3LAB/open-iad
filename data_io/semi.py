import copy
import random


__all__ = ['extract_semi_data']

def extract_semi_data(train_dataset, valid_dataset, anomaly_num=10, anomaly_overlap=False, upper_ratio=0.75):
    valid_sample_nums = [0] + valid_dataset.sample_num_in_task
    valid_sample_indice = valid_dataset.sample_indices_in_task

    # obtain semi label in validation set
    semi_indices = []
    for i in range(valid_dataset.num_task):
        anomaly_index = []
        for k, j in enumerate(range(valid_sample_nums[i], valid_sample_nums[i] + valid_sample_nums[i+1])):
            z = sum(valid_sample_nums[:i+1]) + k
            label = valid_dataset.labels_list[z]
            if label == 1:
                anomaly_index.append(valid_sample_indice[i][k])
        # set semi data to be less than 50 percent of these in test set
        anomaly_num_max = int(len(anomaly_index) * upper_ratio)
        if anomaly_num >= anomaly_num_max:
            anomaly_num = anomaly_num_max
        semi_index = random.sample(anomaly_index, anomaly_num)

        semi_indices.append(semi_index)

    # construct valid_semi_dataset
    valid_semi_dataset = copy.deepcopy(valid_dataset)
    if not anomaly_overlap:
        for task_id in range(valid_dataset.num_task):
            valid_semi_dataset.sample_indices_in_task[task_id] = list(set(valid_sample_indice[task_id]) - set(semi_indices[task_id]))
            valid_semi_dataset.sample_num_in_task[task_id] = len(valid_semi_dataset.sample_indices_in_task[task_id])

    # construct semi_dataset 
    semi_dataset = copy.deepcopy(valid_dataset)
    for task_id in range(valid_dataset.num_task):
        semi_dataset.sample_indices_in_task[task_id] = semi_indices[task_id] 
        semi_dataset.sample_num_in_task[task_id] = len(semi_dataset.sample_indices_in_task[task_id])

    # construct train_semi_dataset
    train_semi_dataset = copy.deepcopy(train_dataset)
    for task_id in range(valid_dataset.num_task):
        for img_id in semi_indices[task_id]:
            train_semi_dataset.imgs_list.append(semi_dataset.imgs_list[img_id]) 
            train_semi_dataset.labels_list.append(semi_dataset.labels_list[img_id]) 
            train_semi_dataset.masks_list.append(semi_dataset.masks_list[img_id]) 
            train_semi_dataset.task_ids_list.append(semi_dataset.task_ids_list[img_id]) 

    for task_id in range(train_dataset.num_task):
        semi_indices = []
        for i in range(semi_dataset.sample_num_in_task[task_id]):
            local_idx = i + len(train_dataset.imgs_list) + sum(semi_dataset.sample_num_in_task[:task_id])
            semi_indices.append(int(local_idx))

        train_semi_dataset.sample_indices_in_task[task_id].extend(semi_indices)
        train_semi_dataset.sample_num_in_task[task_id] += semi_dataset.sample_num_in_task[task_id] 

    return train_semi_dataset, valid_semi_dataset, semi_dataset