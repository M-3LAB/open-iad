import torch
import torch.nn.functional as F

__all__ = ['CumulativeSetFeatures']

class CumulativeSetFeatures(torch.nn.Module):
    def __init__(self, n_channels, n_projections=100, n_quantiles=20, is_projection=True):
        self.n_channels = n_channels
        self.n_projections = n_projections
        self.n_quantiles = n_quantiles
        self.projections = torch.randn(self.n_projections,  self.n_channels, 1)
        self.is_projection = is_projection

    def fit(self, X):
        if self.is_projection:
            a = F.conv1d(X, self.projections).permute((0, 2, 1))
            a = a.reshape((-1, self.n_projections))
        else:
            a = X.permute((0, 2, 1))
            a = a.reshape((a.shape[0]*a.shape[1], -1))
        self.min_vals = torch.quantile(a, 0.01, dim=0)
        self.max_vals = torch.quantile(a, 0.99, dim=0)


    def forward(self, X):
        if self.is_projection:
            a = F.conv1d(X, self.projections)
        else:
            a = X
        cdf = torch.zeros((a.shape[0], a.shape[1], self.n_quantiles))
        set = torch.zeros((a.shape[0], a.shape[1], X.shape[-1], self.n_quantiles,))
        for q in range(self.n_quantiles):
            threshold = self.min_vals + (self.max_vals - self.min_vals) * (q + 1) / (self.n_quantiles + 1)
            set[:, :, :, q] = (a < threshold.unsqueeze(0).unsqueeze(2)).float()
            cdf[:, :, q] = set[:, :, :, q].mean(2)

        set = torch.transpose(set, 2, 1)
        set = set.reshape((X.shape[0], X.shape[-1], -1))
        set = torch.transpose(set, 2, 1).numpy()
        cdf = cdf.reshape((X.shape[0], -1)).numpy()

        return cdf, set