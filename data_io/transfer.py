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
    # source normal dataset is extracted from source train dataset 
    source_train_sample_nums = [0] + source_train_dataset.sample_num_in_task
    source_train_sample_indice = source_train_dataset.sample_indices_in_task

    # source abnormal dataset is extracted from source valid dataset 
    source_valid_sample_nums = [0] + source_valid_dataset.sample_num_in_task
    source_valid_sample_indice = source_valid_dataset.sample_indices_in_task

    # target normal dataset is extracted from target train dataset 
    target_train_sample_nums = [0] + target_train_dataset.sample_num_in_task
    target_train_sample_indice = target_train_dataset.sample_indices_in_task

    # target abnormal dataset is extracted from target valid dataset 
    target_valid_sample_nums = [0] + target_valid_dataset.sample_num_in_task
    target_valid_sample_indice = target_valid_dataset.sample_indices_in_task

    if transfer_type == 'inter_class':
        pass
    elif transfer_type == 'intra_class':
        pass
    else:
        raise NotImplementedError('Transfer Type Has Not Been Implemented')
    
    # construct source anomaly training dataset 
    source_anomaly_indices = []
    for i in range(source_valid_dataset.num_task):
        source_anomaly_index = []
        for k, j in enumerate(range(source_valid_sample_nums[i], source_valid_sample_nums[i] + source_valid_sample_nums[i+1])):
            z = sum(source_valid_sample_nums[:i+1]) + k
            label = source_valid_dataset.label_list[z]
            if label == 1:
                source_anomaly_index.append(source_valid_sample_indice[i][k])
        
        source_anomaly_indices.append(source_anomaly_index)
    
    source_anomaly_dataset = copy.deepcopy(source_valid_dataset)
    for task_id in range(source_valid_dataset.num_task):
        source_anomaly_dataset.sample_indices_in_task[task_id] = source_anomaly_indices[task_id] 
        source_anomaly_dataset.sample_num_in_task[task_id] = len(source_anomaly_dataset.sample_indices_in_task[task_id])

    # construct target anomaly training dataset
    target_anomaly_indices = []
    for i in range(target_valid_dataset.num_task):
        target_anomaly_index = []
        for k, j in enumerate(range(target_valid_sample_nums[i], target_valid_sample_nums[i] + target_valid_sample_nums[i+1])):
            z = sum(target_valid_sample_nums[:i+1]) + k
            label = target_valid_dataset.label_list[z]
            if label == 1:
                target_anomaly_index.append(target_valid_sample_indice[i][k])
        # control target training number 
        target_anomaly_index = random.sample(target_anomaly_index, target_train_num)        
        target_anomaly_indices.append(target_anomaly_index)
    
    target_anomaly_dataset = copy.deepcopy(target_valid_dataset)
    for task_id in range(target_valid_dataset.num_task):
        target_anomaly_dataset.sample_indices_in_task[task_id] = target_anomaly_indices[task_id] 
        target_anomaly_dataset.sample_num_in_task[task_id] = len(target_anomaly_dataset.sample_indices_in_task[task_id])

    # construct target normal training dataset 
    target_train_num_src = target_train_num
    target_fewshot_train_dataset = copy.deepcopy(target_train_dataset)
    for i, num in enumerate(target_train_dataset.sample_num_in_task):
        if target_train_num > num:
            target_train_num = num
        chosen_samples = random.sample(target_fewshot_train_dataset.sample_indices_in_task[i], target_train_num)
        target_fewshot_train_dataset.sample_indices_in_task[i] = chosen_samples
        target_fewshot_train_dataset.sample_num_in_task[i] = target_train_num
        target_train_num = target_train_num_src
    
    # construct total training dataset 
    # total training dataset = source_normal + source_valid + 
    #                          target_normal(few-shot) + target_valid(few-shot)

    train_total_dataset = copy.deepcopy(source_train_dataset)
    #TODO: how to construct train_total_dataset
    # add source abnormal dataset to total training dataset
    for task_id in range(source_train_dataset.num_task):
        for img_id in source_anomaly_indices[task_id]:
            train_total_dataset.imgs_list.append(source_anomaly_dataset.imgs_list[img_id])
            train_total_dataset.labels_list.append(source_anomaly_dataset.labels_list[img_id])
            train_total_dataset.masks_list.append(source_anomaly_dataset.masks_list[img_id])
            train_total_dataset.task_ids_list.append(source_anomaly_dataset.task_ids_list[img_id])
    
    for task_id in range(source_train_dataset.num_task):
        anomaly_indices = [] 
        for i in range(source_anomaly_dataset.sample_num_in_task[task_id]):
            local_idx = i + len(train_total_dataset.imgs_list) + sum(source_anomaly_dataset.sample_num_in_task[:task_id])
            anomaly_indices.append(int(local_idx))
        
        train_total_dataset.sample_indices_in_task[task_id].extend(anomaly_indices)
        train_total_dataset.sample_num_in_task[task_id] += train_total_dataset.sample_num_in_task[task_id] 
    
    return train_total_dataset, source_anomaly_dataset, target_anomaly_dataset, target_fewshot_train_dataset

        

            
    



   

