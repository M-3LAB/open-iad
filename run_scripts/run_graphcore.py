import os

net_list = ['vig_ti_224_gelu', 'vig_s_224_gelu', 'vig_b_224_gelu']
datasets = ['mvtecloco']
num_tasks = [5]
sample_ratio = 0.1
gpu_id = 4
layer_num_1 = 3
layer_num_2 = 4
for net in net_list:
    for dataset, n in zip(datasets, num_tasks):
        train_ids = [i for i in range(n)]
        test_ids = [i for i in range(n)]
        for train_id in train_ids:
            for test_id in test_ids:
                if train_id == test_id:
                    script = 'python3 main.py -p c2d -v -m graphcore -n {} -d {} -tid {} -vid {} -sp {} --layer_num_1 {} --layer_num_2 {} -g {}'.format(
                        net, dataset, train_id, test_id, sample_ratio, layer_num_1, layer_num_2, gpu_id)
                    print(script)
                    os.system(script)