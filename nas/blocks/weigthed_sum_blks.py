# Like FBNet

import numpy as np
import torch.nn as nn
import torch
# import pdb

from .base_blocks import BaseBlock

class WeightedSum(BaseBlock):
  """Sample one or more blocks from
  all blocks, and do forward.
  """

  def __init__(self,
               **kwargs):
    super(WeightedSum, self).__init__(**kwargs)
  
  def forward(self, x, temperature=1.0, **kwargs):
    """Weighted sum forward.

    Parameters
    ----------
    x : tensor
      input
    temperature : float
      default is 1.0, used in gumbel_softmax
    """
    batch_size = x.size()[0]
    weights = self.prob(batch_size=batch_size, temperature=temperature)
    self.latency_loss = self.speed_loss(weights, batch_size)

    tmp = []
    for i, op in enumerate(self.blocks):
      r = op(x)
      w = weights[..., i].reshape((-1, 1, 1, 1))
      res = w * r
      tmp.append(res)
    output = sum(tmp)
    return output, self.latency_loss
