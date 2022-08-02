from cgi import test
import os

train_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
test_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
sample_ratio = 0.001
gpu_id = 1
for train_id in train_ids:
    for test_id in test_ids:
        if train_id == test_id:
            script = 'python3 centralized_training.py --model patchcore2d --chosen-train-task-ids {} --chosen-test-task-id {} --coreset-sampling-ratio {} -g {}'.format(train_id, test_id, sample_ratio, gpu_id)
            os.system(script)