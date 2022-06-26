from torchmetrics import MetricCollection

__all__ = ['OpenADMetricCollection']

class OpenADMetricCollection(MetricCollection):
    """
    Wrap MetricCollction class into openad
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_called = False
        self._threshold = 0.5

    