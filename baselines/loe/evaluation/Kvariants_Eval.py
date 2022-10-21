# Latent Outlier Exposure for Anomaly Detection with Contaminated Data
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import json
import torch
import random
import numpy as np
from loader.LoadData import load_data
from utils import Logger

class KVariantEval:

    def __init__(self, dataset, exp_path, model_configs,contamination,query_num):
        self.num_cls = dataset.num_cls
        self.data_name = dataset.data_name
        self.contamination = contamination
        self.query_num = query_num
        self.model_configs = model_configs

        self._NESTED_FOLDER = exp_path
        self._FOLD_BASE = '_CLS'
        self._RESULTS_FILENAME = 'results.json'
        self._ASSESSMENT_FILENAME = 'assessment_results.json'

    def process_results(self):

        TS_f1s = []
        TS_aps = []
        TS_aucs = []

        assessment_results = {}

        for i in range(self.num_cls):
            try:
                config_filename = os.path.join(self._NESTED_FOLDER, str(i)+self._FOLD_BASE,
                                               self._RESULTS_FILENAME)

                with open(config_filename, 'r') as fp:
                    variant_scores = json.load(fp)
                    ts_f1 = np.array(variant_scores['TS_F1'])
                    ts_auc = np.array(variant_scores['TS_AUC'])
                    ts_ap = np.array(variant_scores['TS_AP'])
                    TS_f1s.append(ts_f1)
                    TS_aucs.append(ts_auc)
                    TS_aps.append(ts_ap)

                assessment_results['avg_TS_f1_' + str(i)] = ts_f1.mean()
                assessment_results['std_TS_f1_' + str(i)] = ts_f1.std()
                assessment_results['avg_TS_ap_' + str(i)] = ts_ap.mean()
                assessment_results['std_TS_ap_' + str(i)] = ts_ap.std()
                assessment_results['avg_TS_auc_' + str(i)] = ts_auc.mean()
                assessment_results['std_TS_auc_' + str(i)] = ts_auc.std()
            except Exception as e:
                print(e)

        TS_f1s = np.array(TS_f1s)
        TS_aps = np.array(TS_aps)
        TS_aucs = np.array(TS_aucs)
        avg_TS_f1 = np.mean(TS_f1s, 0)
        avg_TS_ap = np.mean(TS_aps, 0)
        avg_TS_auc = np.mean(TS_aucs, 0)
        assessment_results['avg_TS_f1_all'] = avg_TS_f1.mean()
        assessment_results['std_TS_f1_all'] = avg_TS_f1.std()
        assessment_results['avg_TS_ap_all'] = avg_TS_ap.mean()
        assessment_results['std_TS_ap_all'] = avg_TS_ap.std()
        assessment_results['avg_TS_auc_all'] = avg_TS_auc.mean()
        assessment_results['std_TS_auc_all'] = avg_TS_auc.std()

        with open(os.path.join(self._NESTED_FOLDER, self._ASSESSMENT_FILENAME), 'w') as fp:
            json.dump(assessment_results, fp, indent=0)

    def risk_assessment(self, experiment_class):

        if not os.path.exists(self._NESTED_FOLDER):
            os.makedirs(self._NESTED_FOLDER)

        for cls in range(self.num_cls):

            folder = os.path.join(self._NESTED_FOLDER, str(cls)+self._FOLD_BASE)
            if not os.path.exists(folder):
                os.makedirs(folder)

            json_results = os.path.join(folder, self._RESULTS_FILENAME)
            if not os.path.exists(json_results):

                self._risk_assessment_helper(cls,  experiment_class, folder)
            else:
                print(
                    f"File {json_results} already present! Shutting down to prevent loss of previous experiments")
                continue

        self.process_results()

    def _risk_assessment_helper(self, cls, experiment_class, exp_path):

        best_config = self.model_configs[0]
        experiment = experiment_class(best_config, exp_path)

        logger = Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')
        # logger = None

        val_auc_list, test_auc_list,test_ap_list,test_f1_list = [], [],[], []
        num_repeat = best_config['num_repeat']

        saved_results = {}
        for i in range(num_repeat):
            print(f'Normal Cls: {cls}')
            torch.cuda.empty_cache()
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(i + 40)
            random.seed(i + 40)
            torch.manual_seed(i + 40)
            torch.cuda.manual_seed(i + 40)
            torch.cuda.manual_seed_all(i + 40)
            trainset, valset, testset = load_data(self.data_name, cls, self.contamination)
            val_auc, test_auc, test_ap,test_f1, test_score = experiment.run_test(trainset,valset,testset,logger,self.contamination,self.query_num)
            print(f'Final training run {i + 1}: {val_auc}, {test_auc,test_ap, test_f1}')

            val_auc_list.append(val_auc)
            test_auc_list.append(test_auc)
            test_ap_list.append(test_ap)
            test_f1_list.append(test_f1)
            if best_config['save_scores']:
                saved_results['scores_'+str(i)] = test_score.tolist()

        if best_config['save_scores']:
            save_path = os.path.join(self._NESTED_FOLDER, str(cls)+self._FOLD_BASE,'scores_labels.json')
            json.dump(saved_results, open(save_path, 'w'))
            # saved_results = json.load(open(save_path))

        if logger is not None:
            logger.log(
                'End of Variant:'+ str(cls) + ' TS f1: ' + str(test_f1_list)+' TS AP: ' + str(test_ap_list)+' TS auc: ' + str(test_auc_list) )
        print('F1:'+str(np.array(test_f1_list).mean())+'AUC:'+str(np.array(test_auc_list).mean()))
        with open(os.path.join(exp_path, self._RESULTS_FILENAME), 'w') as fp:
            json.dump({'best_config': best_config, 'VAL_AUC': val_auc_list,
                       'TS_F1': test_f1_list,'TS_AP': test_ap_list,'TS_AUC': test_auc_list}, fp)


