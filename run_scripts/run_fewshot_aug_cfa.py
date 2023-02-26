import os
import json

datasets = ['imad_hardware_parts']
num_tasks = [21]
fewshots = [1, 2, 4, 8]
sample_ratios = [0.1, 0.1, 0.1, 0.1]
gpu_id = 0
for dataset, n in zip(datasets, num_tasks):
    for fewshot_n, ratio in zip(fewshots, sample_ratios):
        train_ids = [i for i in range(n)]
        sample_ratio = ratio
        for train_id in train_ids:
                f = open('/ssd3/ljq/AD/open-ad/results/fewshot_patchcore.json',)
                data = json.load(f)
                f.close()
                if(train_id<int(data[str(fewshot_n)][dataset])):
                    continue
                else:
                    script = 'python3 centralized_training.py --fewshot -fda --fewshot-exm {} --fewshot-num-dg 4 --model cfa -n net_cfa --coreset-sampling-ratio {} --dataset {} --train-task-id {} --valid-task-id {} -g {}'.format(
                                fewshot_n, ratio, dataset, train_id, train_id, gpu_id)
                    # print(script)
                    os.system(script)
                    f = open('/ssd3/ljq/AD/open-ad/results/fewshot_patchcore.json',)
                    data = json.load(f)
                    f.close()
                    data[str(fewshot_n)][dataset] = train_id+1
                    with open('/ssd3/ljq/AD/open-ad/results/fewshot_patchcore.json', "w") as outfile:
                        json.dump(data, outfile)