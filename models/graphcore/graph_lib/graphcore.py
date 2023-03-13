import torch
import torch.nn as nn

class GraphCore(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config