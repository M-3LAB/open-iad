"""
Compute evaluation metrics for a single experiment.
"""

__author__ = "Paul Bergmann, David Sattlegger"
__copyright__ = "2021, MVTec Software GmbH"

import json
from os import listdir, makedirs, path

import argparse
import numpy as np
import tifffile as tiff
from PIL import Image
from tqdm import tqdm

import generic_util as util
from pro_curve_util import compute_pro
from roc_curve_util import compute_classification_roc


def parse_user_arguments():
    """
    Parse user arguments for the evaluation of a method on the MVTec 3D-AD
    dataset.

    returns:
        Parsed user arguments.
    """
    parser = argparse.ArgumentParser(description="""Parse user arguments.""")

    parser.add_argument('--anomaly_maps_dir',
                        required=True,
                        help="""Path to the directory that contains the anomaly
                                maps of the evaluated method.""")

    parser.add_argument('--dataset_base_dir',
                        required=True,
                        help="""Path to the directory that contains the dataset
                                images of the MVTec 3D-AD dataset.""")

    parser.add_argument('--output_dir',
                        help="""Path to the directory to store evaluation
                                results. If no output directory is specified,
                                the results are not written to drive.""")

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

    parser.add_argument('--evaluated_objects',
                        nargs='+',
                        help="""List of objects to be evaluated. By default,
                                all dataset objects will be evaluated.""",
                        default=util.OBJECT_NAMES)

    args = parser.parse_args()

    # Check that the PRO integration limit is within the valid range.
    assert 0.0 < args.pro_integration_limit <= 1.0

    # Check that the objects to be evaluated are actually available.
    for obj in args.evaluated_objects:
        assert obj in util.OBJECT_NAMES

    # Ensure that the number of evaluated thresholds is at least 2.
    assert args.pro_num_thresholds > 1

    return args


def parse_dataset_files(object_name, dataset_base_dir, anomaly_maps_dir):
    """
    Parse the filenames for one object of the MVTec 3D-AD dataset.

    Args:
        object_name:      Name of the dataset object.
        dataset_base_dir: Base directory of the MVTec 3D-AD dataset.
        anomaly_maps_dir: Base directory where anomaly maps are located.
    """
    assert object_name in util.OBJECT_NAMES

    # Store a list of all ground truth filenames.
    gt_filenames = []

    # Store a list of all corresponding anomaly map filenames.
    prediction_filenames = []

    # Test images are located here.
    test_dir = path.join(dataset_base_dir, object_name, 'test')

    # List all ground truth and corresponding anomaly images.
    for subdir in listdir(str(test_dir)):
        # Ground truth images are located here.
        gt_dir = path.join(test_dir, subdir, 'gt')

        # Add the gt files to the list of all gt filenames.
        gt_filenames.extend(
            [path.join(gt_dir, file)
             for file
             in listdir(gt_dir)
             if path.splitext(file)[1] == '.png'])

        # Get the corresponding filenames of the anomaly images.
        prediction_filenames.extend(
            [path.join(anomaly_maps_dir, object_name, 'test',
                       subdir, path.splitext(file)[0] + '.tiff')
             for file in listdir(gt_dir)])

    print(f"Parsed {len(gt_filenames)} ground truth image files.")

    return gt_filenames, prediction_filenames


def calculate_au_pro_au_roc(gt_filenames,
                            prediction_filenames,
                            integration_limit,
                            num_thresholds):
    """
    Compute the area under the PRO curve for a set of ground truth images
    and corresponding anomaly images. In addition, the function computes the
    area under the ROC curve for image level classification.

    Args:
        gt_filenames:         List of filenames that contain the ground truth
                              images for a single dataset object.
        prediction_filenames: List of filenames that contain the corresponding
                              anomaly images for each ground truth image.
        integration_limit:    Integration limit to use when computing the area
                              under the PRO curve.
        num_thresholds:       Number of thresholds to use to sample the area
                              under the PRO curve.

    Returns:
        au_pro:    Area under the PRO curve computed up to the given integration
                   limit.
        au_roc:    Area under the ROC curve.
        pro_curve: PRO curve values for localization (fpr,pro).
        roc_curve: ROC curve values for image level classifiction (fpr,tpr).

    """
    # Read all ground truth and anomaly images.
    ground_truth = []
    predictions = []

    print("Read ground truth files and corresponding predictions..")
    for (gt_name, pred_name) in tqdm(zip(gt_filenames, prediction_filenames),
                                     total=len(gt_filenames)):
        ground_truth.append(np.asarray(Image.open(gt_name)))
        predictions.append(tiff.imread(pred_name))

    # Derive binary labels for each input image:
    # (0 = anomaly free, 1 = anomalous).
    binary_labels = list(map(lambda x: int(np.any(x > 0)), ground_truth))

    # Compute the PRO curve.
    pro_curve = compute_pro(
        anomaly_maps=predictions,
        ground_truth_maps=ground_truth,
        num_thresholds=num_thresholds)

    # Compute the area under the PRO curve.
    au_pro = util.trapezoid(pro_curve[0], pro_curve[1], x_max=integration_limit)
    au_pro /= integration_limit
    print(f"AU-PRO (FPR limit: {integration_limit}): {au_pro}")

    # Compute the classification ROC curve.
    roc_curve = compute_classification_roc(
        anomaly_maps=predictions,
        scoring_function=np.max,
        ground_truth_labels=binary_labels)

    # Compute the area under the classification ROC curve.
    au_roc = util.trapezoid(roc_curve[0], roc_curve[1])
    print(f"AU-ROC: {au_roc}")

    # Return the evaluation metrics.
    return au_pro, au_roc, pro_curve, roc_curve


def main():
    """
    Calculate the performance metrics for a single experiment on the
    MVTec 3D-AD dataset.
    """
    # Parse user arguments.
    args = parse_user_arguments()

    # Store evaluation results in this dictionary.
    evaluation_dict = dict()

    # Keep track of the mean performance measures.
    mean_au_pro = 0.0
    mean_au_roc = 0.0

    # Evaluate each dataset object separately.
    for obj in args.evaluated_objects:
        print(f"=== Evaluate {obj} ===")
        evaluation_dict[obj] = dict()

        # Parse the filenames of all ground truth and corresponding anomaly
        # images for this object.
        gt_filenames, prediction_filenames = \
            parse_dataset_files(
                object_name=obj,
                dataset_base_dir=args.dataset_base_dir,
                anomaly_maps_dir=args.anomaly_maps_dir)

        # Calculate the PRO and ROC curves.
        au_pro, au_roc, pro_curve, roc_curve = \
            calculate_au_pro_au_roc(
                gt_filenames,
                prediction_filenames,
                args.pro_integration_limit,
                args.pro_num_thresholds)

        evaluation_dict[obj]['au_pro'] = au_pro
        evaluation_dict[obj]['au_roc'] = au_roc

        evaluation_dict[obj]['roc_curve_fpr'] = roc_curve[0]
        evaluation_dict[obj]['roc_curve_tpr'] = roc_curve[1]

        evaluation_dict[obj]['pro_curve_fpr'] = pro_curve[0]
        evaluation_dict[obj]['pro_curve_pro'] = pro_curve[1]

        # Keep track of the mean performance measures.
        mean_au_pro += au_pro
        mean_au_roc += au_roc

        print("")
        print("")

    # Compute the mean of the performance measures.
    evaluation_dict['mean_au_pro'] = mean_au_pro / len(args.evaluated_objects)
    evaluation_dict['mean_au_roc'] = mean_au_roc / len(args.evaluated_objects)

    # If required, write evaluation metrics to drive.
    if args.output_dir is not None:
        makedirs(args.output_dir, exist_ok=True)

        with open(path.join(args.output_dir, 'metrics.json'), 'w') as file:
            json.dump(evaluation_dict, file, indent=4)

        print(f"Wrote metrics to {path.join(args.output_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
