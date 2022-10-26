import copy
import random


__all__ = ['extract_semi_data']

def extract_semi_data(train_dataset, valid_dataset, anomaly_num, anomaly_overlap=False, upper_ratio=0.5):
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
    train_noisy_dataset = copy.deepcopy(train_dataset)
    for i in range(valid_dataset.num_task):
        train_noisy_dataset.sample_indices_in_task[i] = train_dataset.sample_indices_in_task[i] + noisy_indices[i]
        train_noisy_dataset.sample_num_in_task[i] = len(train_noisy_dataset.sample_indices_in_task[i])

    # construct valid_noisy_dataset 
    valid_noisy_dataset = copy.deepcopy(valid_dataset)
    if not anomaly_overlap:
        for i in range(valid_dataset.num_task):
            valid_noisy_dataset.sample_indices_in_task[i] = list(set(valid_sample_indice[i]) - set(noisy_indices[i]))
            valid_noisy_dataset.sample_num_in_task[i] = len(valid_noisy_dataset.sample_indices_in_task[i])

    # construct noisy_dataset 
    noisy_dataset = copy.deepcopy(valid_dataset)
    noisy_dataset.sample_indices_in_task = noisy_indices 
    for i in range(valid_dataset.num_task):
        noisy_dataset.sample_num_in_task[i] = len(noisy_dataset.sample_indices_in_task[i])


    return train_noisy_dataset, valid_noisy_dataset, noisy_dataset
