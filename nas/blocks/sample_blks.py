# Like ProxylessNAS

import numpy as np
import torch.nn as nn
import torch
import pdb

from .base_blocks import BaseBlock
from .utils import scalar2int

class SampleBlock(BaseBlock):
  """Sample one or more blocks from
  all blocks, and do forward.
  """

  def __init__(self,
               **kwargs):
    super(SampleBlock, self).__init__(**kwargs)
    # self.reward_baseline = 0.0
    # self.state_count = 0
  
  # Override
  def prob(self, batch_size):
    """Calculate prob from architecture parameters.
    """
    if batch_size > 1:
      t = self.arch_params.repeat(batch_size, 1)
    else:
      # TODO(ZhouJ) This is inconsistent with base block
      t = self.arch_params
    weight = nn.functional.softmax(t)
    return weight
  
  def speed_loss(self, idx):
    """Override weighted sum.
    This loss need not to be differential because
    of REINFORCE.

    TODO(ZhouJ) Add a baseline to reduce var.
    """
    l = self.speed[idx] #  - self.reward_baseline
    # self.state_count += 1
    # self.reward_baseline = (self.reward_baseline / 
    #                         self.state_count * (self.state_count - 1) -
    #                         self.speed[idx] / self.state_count)
    return l
  
  def forward(self, x, **kwargs):
    """
    TODO(ZhouJ) Only support sample one for now.
    """
    batch_size = x.size()[0]
    weight = self.prob(batch_size=1)

    self.latency_loss = super(SampleBlock, self).speed_loss(weight, 1)
    # sample
    m = torch.distributions.categorical.Categorical(weight)
    action = m.sample()
    choosen_idxs = scalar2int(action)

    output = self.blocks[choosen_idxs](x)

    # REINFORCE
    # reward is minus loss
    p = m.log_prob(action)
    rf_loss = p * self.speed_loss(choosen_idxs)
    return output, rf_loss
