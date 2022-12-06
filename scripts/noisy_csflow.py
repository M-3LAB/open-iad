import os

datasets = ['mvtec2d', 'mpdd']
num_tasks = [15, 6]
gpu_id = 0
for dataset, n in zip(datasets, num_tasks):
    train_ids = [i for i in range(n)]
    for train_id in train_ids:
        script = '/home/ljq/anaconda3/envs/py38torch19/bin/python centralized_training.py --noisy --noisy-ratio 0.1 --noisy-overlap --model csflow -n net_csflow --dataset {} --train-task-id {} --valid-task-id {} -g {}'.format(
        dataset, train_id, train_id, gpu_id)
        # print(script)
        os.system(script)