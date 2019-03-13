# Like darts, snas

import torch.nn as nn
import torch
import torch.nn.functional as F

from .base_blocks import BaseBlock
from ..layers.darts_ops import FactorizedReduce, ReLUConvBN
from ..layers import BASICUNIT

class MixedOp(nn.Module):
  def __init__(self, config, C=None):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for cfg in config:
      op = BASICUNIT[cfg[0]](**cfg[1])
      if 'pool' in cfg[0].lower():
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))

class DAGBlock(BaseBlock):
  """DAG block, every block is a dag with node
  represents data, edge represents operations.
  """
  def __init__(self, C_prev_prev, steps, multiplier, reduction_prev, **kwargs):
    """
    Parameters
    """
    super(DAGBlock, self).__init__(**kwargs)
    self._steps = steps
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, 
            self.out_channels, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, self.out_channels, 
            1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(self.in_channels, self.out_channels,
            1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier
    self.k = sum(1 for i in range(self._steps) for n in range(2+i))
    self.C = self.out_channels

  def build_from_config(self, config):
    """Different from base block.
    See source code of [darts](https://github.com/quark0/darts)
    """
    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        # Every op represnets an edge
        op = MixedOp(config, C=self.C)
        self._ops.append(op)
    self.blocks = self._ops[0]._ops
    self.num_block = len(config)
  
  def init_arch_params(self, init_value=None):
    # TODO Not every layer has a set of parameters
    self._arch_params = nn.Parameter(torch.randn((self.k, self.num_block)), 
                                     requires_grad=True)
    if not init_value is None:
      nn.init.constant_(self._arch_params, init_value)

  def forward(self, s0, s1, temperature=1.0):
    """Every blk is actually an edge.
    """
    batch_size = x.size()[0]
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    weights = F.softmax(self._arch_params)

    speed = torch.Tensor(self.speed, device=weights.device, requires_grad=False)
    cost = torch.matmul(weights, speed).sum()

    states = [s0, s1]
    offset = 0
    for _ in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) 
                  for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1), cost
