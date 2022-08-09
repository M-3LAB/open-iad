from cgi import test
import os

datasets = ['mvtec2d', 'mpdd', 'mvteclogical']
num_tasks = [15, 6, 5]
sample_ratio = 0.0001
gpu_id = 1
for dataset, n in zip(datasets, num_tasks):
    train_ids = [i for i in range(n)]
    test_ids = [i for i in range(n)]
    for train_id in train_ids:
        for test_id in test_ids:
            if train_id == test_id:
                script = 'python3 centralized_training.py --model patchcore2d --dataset {} --chosen-train-task-ids {} --chosen-test-task-id {} --coreset-sampling-ratio {} -g {}'.format(
                    dataset, train_id, test_id, sample_ratio, gpu_id)
                print(script)
                # os.system(script)