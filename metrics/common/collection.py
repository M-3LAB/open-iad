from torchmetrics import MetricCollection

__all__ = ['OpenADMetricCollection']

class OpenADMetricCollection(MetricCollection):
    """
    Wrap MetricCollction class into openad
    """

    def __init__(self):
        super().__init__()
        self._threshold = 0.5