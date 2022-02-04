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

def get_precision_recall():
    """
    Args:
        target: np.ndarray 
        prediction: np.ndarray
    """
    pass