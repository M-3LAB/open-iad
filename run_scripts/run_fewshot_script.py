from cgi import test
import os

fewshots = [1, 3, 5, 10, 15]
train_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
test_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

for fewshot_n in fewshots:
    for train_id in train_ids:
        for test_id in test_ids:
            if train_id == test_id:
                continue
            script = 'python3 centralized_training.py --model patchcore2d --chosen-train-task-ids {} --chosen-test-task-id {} --fewshot --fewshot-exm {} -g {}'.format(train_id, test_id, fewshot_n, 1)
            os.system(script)