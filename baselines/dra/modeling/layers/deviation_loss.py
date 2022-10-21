import torch
import torch.nn as nn

class DeviationLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        confidence_margin = 5.
        ref = torch.normal(mean=0., std=torch.full([5000], 1.)).cuda()
        dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs((confidence_margin - dev).clamp_(min=0.))
        dev_loss = (1 - y_true) * inlier_loss + y_true * outlier_loss
        return torch.mean(dev_loss)
