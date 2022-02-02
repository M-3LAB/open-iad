import torch
import torch.nn as nn
from models.patchcore.knn import *

__all__ = ['PatchCore']

class PatchCore(KNNExtractor):
    def __init__(self):
        super().__init__()
    
    def fit(self, x):
        pass