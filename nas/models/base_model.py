



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
      # register is not necessary, all blk has registered
      # parameter through self.** = Parameter()
      # self.register_parameter(b.name, b.arch_params)

  def forward(self, x, b_input=None, 
              tbs_input=None, head_input=None):
    """Forward

    Parameters
    ----------
    x : torch.tensor
      input
    b_input
      base extra input
    tbs_input
      tbs part extra input
    head_input
      head extra input
    """
    self.batch_size = x.size()[0]
    if b_input is None:
      x = self.base(x)
    else:
      x = self.base(x, b_input)
    
    assert tbs_input is None, 'Not supported for now'
    for i, b in enumerate(self.tbs_blocks):
      if i == 0:
        x, self.blk_loss = self.tbs_blocks(x)
      else:
        x, b_l = self.tbs_blocks(x)
        self.blk_loss += b_l
    
    if head_input is None:
      x = self.head(x)
    else:
      x = self.head(x, head_input)

  def head_loss_(self, output, target):
    return self.head.loss_(output, target)
  
  def speed_test(self, x, b_input=None, 
                 tbs_input=None, head_input=None,
                 device='cuda'):
    """Measure speed for tbs blocks.
    """
    if b_input is None:
      x = self.base(x)
    else:
      x = self.base(x, b_input)
    
    for blk in self.tbs_blocks:
      x = blk.speed_test(x, device=device)
    
  def loss_(self, x, y):
    """Calculate loss and return it.

    Under most circumstance, you want to override this.
    """
    raise NotImplementedError()




