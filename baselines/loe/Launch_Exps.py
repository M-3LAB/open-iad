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

import argparse
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from config.base import Grid, Config
from evaluation.Experiments import runExperiment
from evaluation.Kvariants_Eval import KVariantEval

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file', default='config_cifar10.yml')
    parser.add_argument('--dataset-name', dest='dataset_name', default='cifar10')
    parser.add_argument('--contamination', type=float, default=0.1)
    parser.add_argument('--query_num', type=int, default=0) # for active anomaly detection
    return parser.parse_args()

def EndtoEnd_Experiments(config_file, dataset_name,contamination,query_num):

    model_configurations = Grid(config_file, dataset_name)
    model_configuration = Config(**model_configurations[0])
    dataset =model_configuration.dataset
    result_folder = model_configuration.result_folder+model_configuration.exp_name

    exp_path = os.path.join(result_folder,f'{contamination}_{model_configuration.train_method}')
    risk_assesser = KVariantEval(dataset, exp_path, model_configurations,contamination,query_num)

    risk_assesser.risk_assessment(runExperiment)

if __name__ == "__main__":
    args = get_args()
    config_file = 'config_files/'+args.config_file
    EndtoEnd_Experiments(config_file, args.dataset_name,args.contamination,args.query_num)
