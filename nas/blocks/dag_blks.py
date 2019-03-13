# Like darts, snas

import torch.nn as nn
import torch

from .base_blocks import BaseBlock

class DAGBlock(BaseBlock):
  """DAG block, every block is a dag with node
  represents data, edge represents operations.
  """
  def __init__(self, steps=4, multiplier=4, **kwargs):
    """
    Parameters
    """
    self._steps = steps
    self._multiplier = multiplier
    super(DAGBlock, self).__init__(**kwargs)


  def forward(self, s0, s1, temperature=1.0):
    """Every blk is actually an edge.
    """
    batch_size = x.size()[0]
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    weights = self.prob(batch_size=batch_size, temperature=temperature)

    states = [s0, s1]
    offset = 0
    for _ in range(self._steps):
      s = sum(self.blocks[offset+j](h, weights[offset+j]) 
                  for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)
