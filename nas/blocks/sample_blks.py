# Like ProxylessNAS

import numpy as np
import torch.nn as nn

from base_blocks import BaseBlock
from utils import scalar2int

class SampleBlock(BaseBlock):
  """Sample one or more blocks from
  all blocks, and do forward.
  """

  def __init__(self,
               **kwargs):
    
    super(SampleBlocks, self).__init__(**kwargs)
  
  # Override
  def prob(self, batch_size, temperature):
    """Calculate prob from architecture parameters.
    """
    t = self.arch_params.repeat(batch_size, 1)
    weight = nn.functional.softmax(t)
    return weight

  def forward(self, x):
    """
    TODO(ZhouJ) Only support sample one for now.
    """
    m = torch.distributions.categorical.Categorical(self.prob())
    action = m.sample(1)
    choosen_idxs = scalar2int(action)
    return self.blocks[choosen_idxs](x), m.log_prob(action)

