"""
Run the evaluation script for multiple experiments.

This is a wrapper around evaluate_experiments.py which is called once for each
experiment specified in the config file passed to this script.
"""

__author__ = "Paul Bergmann, David Sattlegger"
__copyright__ = "2021, MVTec Software GmbH"

import argparse
import json
import subprocess
from os import path


def parse_user_arguments():
    """
    Parse user arguments.

    Returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="""Parse user arguments.""")

    parser.add_argument('--experiment_configs',
                        default='experiment_configs.json',
                        help="""Path to the config file that contains the
                                locations of all experiments that should be
                                evaluated.""")

    parser.add_argument('--dataset_base_dir',
                        required=True,
                        help="""Path to the directory that contains the dataset
                                images of the MVTec 3D-AD dataset.""")

    parser.add_argument('--output_dir',
                        default='metrics/',
                        help="""Path to write evaluation results to.""")

    parser.add_argument('--dry_run',
                        choices=['True', 'False'],
                        default='False',
                        help="""If set to 'True', the script is run without
                                perfoming the actual evalutions. Instead, the
                                experiments to be evaluated are simply printed
                                to the standard output.""")

    parser.add_argument('--pro_integration_limit',
                        type=float,
                        default=0.3,
                        help="""Integration limit to compute the area under
                                the PRO curve. Must lie within the interval
                                of (0.0, 1.0].""")

    parser.add_argument('--pro_num_thresholds',
                        type=int,
                        default=100,
                        help="""Number of thresholds to use to sample the
                                area under the PRO curve.""")

    return parser.parse_args()


def main():
    """
    Run the evaluation script for multiple experiments.
    """
    # Parse user arguments.
    args = parse_user_arguments()

    # Read the experiment configurations to be evaluated.
    with open(args.experiment_configs) as file:
        experiment_configs = json.load(file)

    # Call the evaluation script for each experiment separately.
    for experiment_id in experiment_configs['anomaly_maps_dirs']:
        print(f"=== Evaluate experiment: {experiment_id} ===\n")

        # Anomaly maps for this experiment are located in this directory.
        anomaly_maps_dir = path.join(
            experiment_configs['exp_base_dir'],
            experiment_configs['anomaly_maps_dirs'][experiment_id])

        # Set up python call for the evaluation script.
        call = ['python', 'evaluate_experiment.py',
                '--anomaly_maps_dir', anomaly_maps_dir,
                '--dataset_base_dir', args.dataset_base_dir,
                '--output_dir', path.join(args.output_dir, experiment_id),
                '--pro_integration_limit', str(args.pro_integration_limit),
                '--pro_num_thresholds', str(args.pro_num_thresholds)]

        # Run evaluation script.
        if args.dry_run == 'False':
            subprocess.run(call, check=True)
        else:
            print(f"Would call: {call}\n")


if __name__ == "__main__":
    main()
