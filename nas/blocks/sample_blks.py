# Like ProxylessNAS

import numpy as np
import torch.nn as nn
import torch

from .base_blocks import BaseBlock
from .utils import scalar2int

class SampleBlock(BaseBlock):
  """Sample one or more blocks from
  all blocks, and do forward.
  """

  def __init__(self,
               **kwargs):
    super(SampleBlock, self).__init__(**kwargs)
  
  # Override
  def prob(self, batch_size):
    """Calculate prob from architecture parameters.
    """
    if batch_size > 1:
      t = self.arch_params.repeat(batch_size, 1)
    else:
      t = self.arch_params
    weight = nn.functional.softmax(t)
    return weight
  
  def speed_loss(self, idx):
    """Override weighted sum.
    This loss need not to be differential because
    of REINFORCE.
    """
    return self.speed[idx]
  
  def forward(self, x):
    """
    TODO(ZhouJ) Only support sample one for now.
    """
    batch_size = x.size()[0]
    weight = self.prob(batch_size=1)
    m = torch.distributions.categorical.Categorical(weight)
    action = m.sample()
    choosen_idxs = scalar2int(action)
    output = self.blocks[choosen_idxs](x)
    p = m.log_prob(action)

    # REINFORCE
    # reward is minus loss
    rf_loss = p * self.speed_loss(choosen_idxs)
    return output, rf_loss

