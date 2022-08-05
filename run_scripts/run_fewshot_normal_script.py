from cgi import test
import os

datasets = ['mvtec2d', 'mpdd', 'mvteclogical']
num_tasks = [15, 6, 5]
fewshots = [1, 2, 4, 8]
sample_ratios = [1., 1., 0.1, 0.01]
gpu_id = 4
das = [0, 1]
fas = [0, 1]
for dataset, n in zip(datasets, num_tasks):
    for fewshot_n, ratio in zip(fewshots, sample_ratios):
        train_ids = [i for i in range(n)]
        test_ids = [i for i in range(n)]
        sample_ratio = 0.01
        for train_id in train_ids:
            for test_id in test_ids:
                for da in das:
                    for fa in fas:
                        if train_id == test_id:
                            script = 'python3 centralized_training.py --fewshot-normal --model patchcore2d --dataset {} --chosen-train-task-ids {} --chosen-test-task-id {} --fewshot-exm {} --coreset-sampling-ratio {} -g {}'.format(
                                        dataset, train_id, test_id, fewshot_n, sample_ratio, gpu_id)
                            if da:
                                script = '{} -da'.format(script)
                            if fa:
                                script = '{} -fa'.format(script)
                            print(script)
                            os.system(script)