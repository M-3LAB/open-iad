import numpy as np

class SamplingMethod(object):

  def __init__(self, x, y, seed, **kwargs):
    self.x = x
    self.y = y
    self.seed = seed

  def flatten_X(self):
    shape = self.x.shape
    flat_x = self.x
    if len(shape) > 2:
      flat_x = np.reshape(self.x, (shape[0], np.product(shape[1:])))
    return flat_x

  def select_batch(self):
      pass

  def to_dict(self):
    return None