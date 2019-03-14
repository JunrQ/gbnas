# Like darts, snas

import torch.nn as nn
import torch
import torch.nn.functional as F

from .base_blocks import BaseBlock
from ..layers.darts_ops import FactorizedReduce, ReLUConvBN
from ..layers import BASICUNIT
from .utils import measure_speed

class MixedOp(nn.Module):
  def __init__(self, config, C, stride=1, name='mixedop'):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.name = name
    for cfg in config:
      if 'stride' in cfg[1].keys():
        cfg[1]['stride'] = stride
      op = BASICUNIT[cfg[0]](**cfg[1])
      if 'pool' in cfg[0].lower():
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)
  
  def speed_test(self, x, device='cuda', times=200, verbose=True):
    self.speed = [] # * len(self._ops)
    msg = ''
    for i, b in enumerate(self._ops):
      s, o = measure_speed(b, x, device, times)
      msg += "%s %d block speed %.5f ms\n" % (self.name, i, s)
      self.speed.append(s)
    if verbose:
      print(msg)
    return o

  def forward(self, x, weights):
    if isinstance(self.speed, list):
      self.speed = torch.tensor(self.speed, requires_grad=False)
      self.speed = self.speed.to(x.device)
    self.cost = torch.dot(self.speed, weights)
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
        reduction = (self.stride == 2)
        stride = 2 if reduction and j < 2 else 1
        name = self.name + '_node_%d_node_%d' % (j, i+2)
        op = MixedOp(config, C=self.C, stride=stride, name=name)
        self._ops.append(op)
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
    # batch_size = s0.size()[0]
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    weights = F.softmax(self._arch_params)

    states = [s0, s1]
    offset = 0
    time_cost = 0
    for _ in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) 
                  for j, h in enumerate(states))
      time_cost += sum(self._ops[offset+j].cost for j in range(len(states)))

      offset += len(states)
      states.append(s)
    return torch.cat(states[-self._multiplier:], dim=1), time_cost

  def speed_test(self, s0, s1, device='cuda', times=200, verbose=True):
    print(self.name, 's0', s0.size(), 's1', s1.size())
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for _ in range(self._steps):
      s = sum(self._ops[offset+j].speed_test(h, device=device, 
                                             times=times, verbose=verbose) 
                for j, h in enumerate(states))
      offset += len(states)
      states.append(s)
    return torch.cat(states[-self._multiplier:], dim=1)
