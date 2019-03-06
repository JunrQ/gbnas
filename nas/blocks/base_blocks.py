
import numpy as np
import torch.nn as nn
import torch

from ..layers import BASICUNIT
from .utils import measure_speed

class BaseBlock(nn.Module):
  """Base class for TBS(to be search) blocks.

  **Note:** forward should return (output, extra_loss)
  """

  def __init__(self, in_channels,
               out_channels,
               name,
               stride=1,
               **kwargs):
    """
    Parameters
    ----------
    in_channels : int
      num of input channels
    out_channels : int
      num of output channels
    
    """
    super(BaseBlock, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self._mod_params = None
    self._arch_params = None
    self.name = name
    self.stride = stride
  
  def init_arch_params(self, init_value=1.0):
    """Initilize architecture parameters and register
    them into self.parameters() if they are not.
    Register is important for multi-gpus training.
    """
    self.num_blocks = len(self.blocks)
    self._arch_params = nn.Parameter(torch.ones((self.num_block, )), 
                         requires_grad=True)
    nn.init.constant_(self._arch_params, init_value)
  
  def prob(self, batch_size, temperature=1.0):
    """Calculate prob from architecture parameters.
    """
    t = self.arch_params.repeat(batch_size, 1)
    weight = nn.functional.gumbel_softmax(t,
                                temperature)
    return weight

  @property
  def arch_params(self):
    return self._arch_params

  @property
  def model_params(self):
    # TODO(ZhouJ) arch_params is type tensor, but model_params is list,
    if self._mod_params is None:
      self._mod_params = []
      for n, p in self.named_parameters():
        if '_arch_params' not in n:
          self._mod_params.append(p)
    return self._mod_params

  def build_from_config(self, config):
    """Build from config.
    Config should be a list with every elements(TBS layer) should also be a list,
    with each elements represents one block.
    Each block should be a list with two element, 
    the first represent the type, 
    the second represent the kwargs.

    e.g.
    """
    self.blocks = []
    for blk in config:
      t_blk = BASICUNIT[blk[0]](**blk[1])
      self.blocks.append(t_blk)
    self.blocks = nn.ModuleList(self.blocks)
    self.num_block = len(self.blocks)
  
  def speed_test(self, x, device='cuda', times=200,
                 verbose=True):
    """Speed test.

    TODO(ZhouJ) output is the last blk's output
    """
    self.speed = []
    msg = ''
    for i, b in enumerate(self.blocks):
      s, o = measure_speed(b, x, device, times)
      msg += "%s %d block speed %.5f ms\n" % (self.name, i, s)
      self.speed.append(s)
    if verbose:
      print(msg)
    return o

  def speed_loss(self, weight, batch_size):
    """Override this method if you need a different
    speed loss.
    The default is weighted sum of all blocks.
    """
    assert hasattr(self, 'speed'), 'Make sure you run speed_test before'
    if isinstance(self.speed, list):
      self.speed = torch.tensor(self.speed).to(weight.device)
    if batch_size > 1:
      s = self.speed.repeat(batch_size, 1)
    else:
      s = self.speed
    l = torch.sum(torch.mul(weight, s))
    return l
