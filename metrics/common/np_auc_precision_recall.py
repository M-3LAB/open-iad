from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score 
import numpy as np

__all__ = ['np_get_auroc', 'np_get_precision_recall', 'np_get_ap',
           'np_get_aupro']

def np_get_auroc(target, prediciton):
    """
    Args:
        target: np.ndarray 
        prediction: np.ndarray
    """
    score = roc_auc_score(target, prediciton)
    return score

def np_get_precision_recall(target, prediction):
    """
    Args:
        target: np.ndarray 
        prediction: np.ndarray
    """
    precision, recall, thresholds = precision_recall_curve(target.flatten(), prediction.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    idx = np.argmax(f1)
    return precision[idx], recall[idx], thresholds[idx]

def np_get_ap(target, prediction):
    ap = average_precision_score(target, prediction) 
    return ap

def np_get_aupro(masks, amaps, num_th):
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """
    pass