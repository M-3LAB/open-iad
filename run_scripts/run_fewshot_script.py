from cgi import test
import os

fewshots = [5]
train_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
test_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
sample_ratio = 1
gpu_id = 7
for fewshot_n in fewshots:
    for train_id in train_ids:
        for test_id in test_ids:
            if train_id == test_id:
                continue
            script = 'python3 centralized_training.py --fewshot-normal --model patchcore2d --chosen-train-task-ids {} --chosen-test-task-id {} --fewshot-exm {} --coreset-sampling-ratio {} -g {}'.format(
                    train_id, test_id, fewshot_n, sample_ratio, gpu_id)
            os.system(script)