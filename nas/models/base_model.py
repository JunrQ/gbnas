



import torch
import torch.nn as nn

class BaseModel(nn.Module):
  """Base class for building neural network.
  """

  def __init__(self, base,
               tbs_blocks,
               head):
    """
    Parameters
    ----------
    base : 

    tbs_blocks : list

    head : 

    """
    super(BaseModel, self).__init__()

    self.base = base
    if isinstance(tbs_blocks, list):
      self.tbs_blocks = nn.ModuleList(tbs_blocks)
    elif isinstance(tbs_blocks, nn.Module):
      pass
    else:
      raise TypeError("tbs_blocks should be type list or nn.Module")
    self.head = head

    self.arch_params = []
    for b in self.tbs_blocks:
      self.arch_params.append(b.arch_params)
      self.register_parameter(b.name, b.arch_params)
  

  def forward(self, x):

    x = self.base(x)
    x = self.tbs_blocks(x)
    x = self.head(x)

  def loss(self, output, target):
    return self.head.loss(output, target)



