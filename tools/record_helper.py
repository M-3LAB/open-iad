import os
from rich import print

__all__ = ['RecordHelper']

class RecordHelper():
    def __init__(self, config):
        self.config = config

    def update(self, config):
        self.config = config
    
    def printer(self, info):
        print(info)

    def paradigm_name(self):
        if self.config['vanilla']:
            return 'vanilla'
        if self.config['semi']:
            return 'semi'
        if self.config['continual']:
            return 'continual'
        if self.config['fewshot']:
            return 'fewshot'
        
        return 'unknown'

    def record_result(self, paradim, result):
        save_dir = '{}/benchmark/{}/{}/{}/{}'.format(self.config['work_dir'], paradim, self.config['dataset'],
                                                                 self.config['model'], self.config['train_task_id_tmp'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # if paradim == 'vanilla':
        save_path = save_dir + '/result.txt'

        if paradim == 'semi':
            save_path = '{}/result_{}.txt'.format(save_dir, self.config['semi_anonmaly_num'])

        if paradim == 'fewshot':
            save_path = '{}/result_{}.txt'.format(save_dir, self.config['fewshot_exm'])

        if paradim == 'noisy':
            save_path = '{}/result_{}.txt'.format(save_dir, self.config['noisy_ratio'])

        if paradim == 'continual':
            save_path = '{}/result_{}.txt'.format(save_dir, self.config['valid_task_id_tmp'])
        
        with open(save_path, 'a') as f:
            print(result, file=f) 