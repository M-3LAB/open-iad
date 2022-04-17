"""
Print the key metrics of multiple experiments to the standard output.
"""

__author__ = "Paul Bergmann, David Sattlegger"
__copyright__ = "2021, MVTec Software GmbH"

import argparse
import json
import os
from os.path import join

import numpy as np
from tabulate import tabulate

from generic_util import OBJECT_NAMES


def parse_user_arguments():
    """
    Parse user arguments.

    Returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="""Parse user arguments.""")

    parser.add_argument('--metrics_folder',
                        default="./metrics/",
                        help="""Path to the folder that contains the evaluation
                                results.""")

    return parser.parse_args()


def extract_table_rows(metrics_folder, metric):
    """
    Extract all rows to create a table that displays a given metric for each
    evaluated method.

    Args:
        metrics_folder: Base folder that contains evaluation results.
        metric:         Name of the metric to be extracted. Choose between
                        'au_pro' for localization and
                        'au_roc' for classification.

    Returns:
        List of table rows. Each row contains the method name and the extracted
            metrics for each evaluated object as well as the mean performance.
    """
    assert metric in ['au_pro', 'au_roc']

    # Iterate each evaluated method.
    method_ids = os.listdir(metrics_folder)
    rows = []
    for method_id in method_ids:

        # Each row starts with the name of the method.
        row = [method_id]

        # Open the metrics file.
        with open(join(metrics_folder, method_id, 'metrics.json')) as file:
            metrics = json.load(file)

        # Parse performance metrics for each evaluated object if available.
        for obj in OBJECT_NAMES:
            if obj in metrics:
                row.append(np.round(metrics[obj][metric], decimals=3))
            else:
                row.append(np.round(metrics[obj]["-"], decimals=3))

        # Parse mean performance.
        row.append(np.round(metrics['mean_' + metric], decimals=3))
        rows.append(row)

    return rows


def main():
    """
    Print the key metrics of multiple experiments to the standard output.
    """
    # Parse user arguments.
    args = parse_user_arguments()

    # Create the table rows. One row for each experiment.
    rows_pro = extract_table_rows(args.metrics_folder, 'au_pro')
    rows_roc = extract_table_rows(args.metrics_folder, 'au_roc')

    # Print localization result table.
    print("\nAU PRO (localization)")
    print(
        tabulate(
            rows_pro, headers=['Method'] + OBJECT_NAMES + ['Mean'],
            tablefmt='fancy_grid'))

    # Print classification result table.
    print("\nAU ROC (classification)")
    print(
        tabulate(
            rows_roc, headers=['Method'] + OBJECT_NAMES + ['Mean'],
            tablefmt='fancy_grid'))


if __name__ == "__main__":
    main()
