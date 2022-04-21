"""
Utility functions that compute a ROC curve and integrate its area from a set
of anomaly maps and corresponding ground truth classification labels.
"""
import numpy as np

from generic_util import trapezoid, generate_toy_dataset


def compute_classification_roc(
        anomaly_maps,
        scoring_function,
        ground_truth_labels):
    """
    Compute the ROC curve for anomaly classification on the image level.

    Args:
        anomaly_maps:        List of anomaly maps (2D numpy arrays) that contain
                             a real-valued anomaly score at each pixel.
        scoring_function:    Function that turns anomaly maps into a single
                             real valued anomaly score.

        ground_truth_labels: List of integers that indicate the ground truth
                             class for each input image. 0 corresponds to an
                             anomaly-free sample while a value != 0 indicates
                             an anomalous sample.

    Returns:
        fprs: List of false positive rates.
        tprs: List of correspoding true positive rates.
    """
    assert len(anomaly_maps) == len(ground_truth_labels)

    # Compute the anomaly score for each anomaly map.
    anomaly_scores = map(scoring_function, anomaly_maps)

    # Sort samples by anomaly score. Keep track of ground truth label.
    sorted_samples = \
        sorted(zip(anomaly_scores, ground_truth_labels), key=lambda x: x[0])

    # Compute the number of OK and NOK samples from the ground truth.
    ground_truth_labels_np = np.array(ground_truth_labels)
    num_nok = ground_truth_labels_np[ground_truth_labels_np != 0].size
    num_ok = ground_truth_labels_np[ground_truth_labels_np == 0].size

    # Initially, every NOK sample is correctly classified as anomalous
    # (tpr = 1.0), and every OK sample is incorrectly classified as anomalous
    # (fpr = 1.0).
    fprs = [1.0]
    tprs = [1.0]

    # Keep track of the current number of false and true positive predictions.
    num_fp = num_ok
    num_tp = num_nok

    # Compute new true and false positive rates when successively increasing
    # the threshold.
    for _, label in sorted_samples:
        if label == 0:
            num_fp -= 1
        else:
            num_tp -= 1

        fprs.append(num_fp / num_ok)
        tprs.append(num_tp / num_nok)

    # Return (FPR, TPR) pairs in increasing order.
    fprs = fprs[::-1]
    tprs = tprs[::-1]

    return fprs, tprs


def main():
    """
    Compute the area under the ROC curve for a toy dataset and an algorithm
    that randomly assigns anomaly scores to each image pixel.
    """
    # Fix a random seed for reproducibility.
    np.random.seed(42)

    # Generate a toy dataset.
    anomaly_maps, _ = generate_toy_dataset(
        num_images=10000, image_width=30, image_height=30, gt_size=0)

    # Assign a random classification label to each image.
    labels = np.random.randint(2, size=len(anomaly_maps))

    # Compute the ROC curve.
    all_fprs, all_tprs = compute_classification_roc(anomaly_maps=anomaly_maps,
                                                    scoring_function=np.max,
                                                    ground_truth_labels=labels)

    # Compute the area under the ROC curve.
    au_roc = trapezoid(all_fprs, all_tprs)
    print(f"AU-ROC: {au_roc}")


if __name__ == "__main__":
    main()
