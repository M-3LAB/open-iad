import os

datasets = ['mvtec2d', 'visa', 'btad', 'mtd', 'mpdd']
fewshots = [1, 2, 4, 8]
num_tasks = [15, 12, 3, 1, 6]
sample_ratios = [0.1, 0.1, 0.1, 0.1]
aug_type = ['normal', 'rotation', 'scale', 'translate', 'flip', 'color_jitter', 'perspective']
# datasets = ['mvtec2d']
# fewshots = [1]
# num_tasks = [1]
# sample_ratios = [0.1]
# aug_type = ['normal', 'rotation']
gpu_id = 6
result_dir = '/ssd2/m3lab/usrs/zsh/open-ad/work_dir/benchmark/fewshot'
for dataset, n in zip(datasets, num_tasks):
    for fewshot_n, ratio in zip(fewshots, sample_ratios):
        train_ids = [i for i in range(n)]
        sample_ratio = ratio
        for train_id in train_ids:
            for aug in aug_type:
                txt_path = os.path.join(result_dir, dataset, 'favae', 'task_{}'.format(train_id),
                                        'result_{}_{}_shot.txt'.format(aug, fewshot_n))
                if os.path.exists(txt_path):
                    continue
                if aug == 'normal':
                    script = 'python3 main.py -p c2d -f --fewshot-exm {} -m favae -n net_favae -d {} -tid {} -vid {} -g {}'.format(fewshot_n, dataset, train_id, train_id, gpu_id)
                else:
                    script = 'python3 main.py -p c2d -f -fda --fewshot-exm {} -m favae -n net_favae -d {} -tid {} -vid {} -g {} -fnd 3 -fat {}'.format(fewshot_n, dataset, train_id, train_id, gpu_id, aug)
                # print(script)
                os.system(script)