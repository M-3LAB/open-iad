import torch
import numpy as np

__all__ = ['min_max_normalize']

def min_max_normalize(targets, threshold, min_val, max_val):
    """Apply min-max normalization for the target value

    Args:
        targets (np.ndarray or Tensor): _description_
        threshold (np.ndarray or Tensor): _description_
        min_val (float): _description_
        max_val (float): _description_
    """

    normalized = ((targets - threshold) / (max_val - min_val)) + 0.5
    if isinstance(targets, (np.ndarray, np.float32)):
        normalized = np.minimum(normalized, 1)
        normalized = np.maximum(normalized, 0)
    elif isinstance(targets, torch.Tensor):
        normalized = torch.minimum(normalized, torch.tensor(1))  # pylint: disable=not-callable
        normalized = torch.maximum(normalized, torch.tensor(0))  # pylint: disable=not-callable
    else:
        raise ValueError(f"Targets must be either Tensor or Numpy array. Received {type(targets)}")
    return normalized
