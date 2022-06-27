import torch
from torch import Tensor
from torchmetrics import Metric

__all__= ['AnomalyScoreDistribution']

class AnomalyScoreDistribution(Metric):
    """Mean and standard deviation of the anomaly scores of normal training data."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.anomaly_maps = []
        self.anomaly_scores = []

        self.add_state("image_mean", torch.empty(0), persistent=True)
        self.add_state("image_std", torch.empty(0), persistent=True)
        self.add_state("pixel_mean", torch.empty(0), persistent=True)
        self.add_state("pixel_std", torch.empty(0), persistent=True)

        self.image_mean = torch.empty(0)
        self.image_std = torch.empty(0)
        self.pixel_mean = torch.empty(0)
        self.pixel_std = torch.empty(0)

    # pylint: disable=arguments-differ
    def update(self, anomaly_scores, anomaly_maps):
        """Update the precision-recall curve metric."""
        if anomaly_maps is not None:
            self.anomaly_maps.append(anomaly_maps)
        if anomaly_scores is not None:
            self.anomaly_scores.append(anomaly_scores)

    def compute(self):
        """Compute stats."""
        anomaly_scores = torch.hstack(self.anomaly_scores)
        anomaly_scores = torch.log(anomaly_scores)

        self.image_mean = anomaly_scores.mean()
        self.image_std = anomaly_scores.std()

        if self.anomaly_maps:
            anomaly_maps = torch.vstack(self.anomaly_maps)
            anomaly_maps = torch.log(anomaly_maps).cpu()

            self.pixel_mean = anomaly_maps.mean(dim=0).squeeze()
            self.pixel_std = anomaly_maps.std(dim=0).squeeze()

        return self.image_mean, self.image_std, self.pixel_mean, self.pixel_std