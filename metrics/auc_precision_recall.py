from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import numpy as np

__all__ = ['get_auc', 'get_precision_recall']

def get_auc(target, prediciton):
    """
    Args:
        target: np.ndarray 
        prediction: np.ndarray
    """
    score = roc_auc_score(target, prediciton)
    return score

def get_precision_recall(target, prediction):
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