import os
import json

datasets = ['mvtec2d', 'visa', 'btad', 'mtd', 'mpdd']
fewshots = [2, 4, 8]
num_tasks = [15, 12, 3, 1, 6]
sample_ratios = [0.1, 0.1, 0.1, 0.1]
aug_type = ['normal', 'rotation', 'scale', 'translate', 'flip', 'color_jitter', 'perspective']
# datasets = ['mvtec2d']
# fewshots = [2]
# num_tasks = [1]
# sample_ratios = [0.1]
# aug_type = ['normal', 'rotation']
gpu_id = 5
json_file = '/ssd2/m3lab/usrs/zsh/open-ad/run_scripts/fewshot_spade.json'
for dataset, n in zip(datasets, num_tasks):
    for fewshot_n, ratio in zip(fewshots, sample_ratios):
        train_ids = [i for i in range(n)]
        sample_ratio = ratio
        for train_id in train_ids:
            f = open(json_file)
            data = json.load(f)
            f.close()
            if train_id < int(data[str(fewshot_n)][dataset]):
                continue
            else:
                for aug in aug_type:
                    if aug == 'normal':
                        if fewshot_n < 8:
                            continue
                        script = 'python3 main.py -p c2d -f --fewshot-exm {} -m spade -n resnet18 -d {} -tid {} -vid {} -g {}'.format(fewshot_n, dataset, train_id, train_id, gpu_id)
                    else:
                        script = 'python3 main.py -p c2d -f -fda --fewshot-exm {} -m spade -n resnet18 -d {} -tid {} -vid {} -g {} -fnd 3 -fat {}'.format(fewshot_n, dataset, train_id, train_id, gpu_id, aug)
                    # print(script)
                    os.system(script)
                f = open(json_file)
                data = json.load(f)
                f.close()
                data[str(fewshot_n)][dataset] = train_id + 1
                with open(json_file, "w") as outfile:
                    json.dump(data, outfile)