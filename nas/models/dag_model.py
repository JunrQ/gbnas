import torch
from .classify_model import ClassificationModel

class DAGModel(ClassificationModel):
  """Base model for classification.

  TODO Architecture paramters should be shared among
  different layers.
  """

  def __init__(self, base,
               tbs_blocks,
               head):
    super(DAGModel, self).__init__(base=base,
          tbs_blocks=tbs_blocks, head=head)

  def forward(self, x, y, base_input=None, 
              tbs_input=None, head_input=None,
              mode='w'):
    """Forward

    Parameters
    ----------
    x : torch.tensor
      input
    base_input
      base extra input
    tbs_input
      tbs part extra input
    head_input
      head extra input
    """
    self.batch_size = x.size()[0]

    # base forward
    if base_input is None:
      x = self.base(x)
    else:
      x = self.base(x, base_input)

    s0 = s1 = x
    # tbs forward
    if tbs_input is None:
      tbs_input = {}
    for i, b in enumerate(self.tbs_blocks):
      if i == 0:
        s0, (s1, self.blk_loss) = s1, b(s0, s1, **tbs_input)
      else:
        s0, (s1, b_l) = s1, b(s0, s1, **tbs_input)
        self.blk_loss += b_l
    # head forward
    x = s1
    if head_input is None:
      x = self.head(x)
    else:
      x = self.head(x, head_input)
    l, ce = self.loss_(x, y)
  
    pred = torch.argmax(x, dim=1)
    acc = torch.sum(pred == y).float() / self.batch_size
    return l, ce, acc
  
  def speed_test(self, x, base_input=None, 
                 tbs_input=None, head_input=None,
                 device='cuda', verbose=True):
    if verbose:
      print("Doing speed test")
    # base forward
    self.to(device)
    x = x.to(device)
    if base_input is None:
      x = self.base(x)
    else:
      x = self.base(x, base_input)
    s0 = s1 = x

    for i, b in enumerate(self.tbs_blocks):
      s0, s1 = s1, b.speed_test(s0, s1, 
          device=device, verbose=verbose)
