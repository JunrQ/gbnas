import torch
from .base_model import BaseModel

class ClassificationModel(BaseModel):
  """Base model for classification.
  """

  def __init__(self, base,
               tbs_blocks,
               head):
    """
    """
    super(ClassificationModel, self).__init__(base=base,
          tbs_blocks=tbs_blocks, head=head)

  @property
  def ce(self):
    """Get cross entropy.
    """
    return self.head.ce_loss
  
  @property
  def acc(self):
    """Get accuracy.
    """
    return self.head.acc
  
  def register_loss_func(self, func):
    self.loss_func = func

  def loss_(self, x, y, mode=None):
    """Calculate loss and return it.

    Under most circumstance, you want to override this.
    
    Like FBNet, total_loss = ce + alpht * lat_loss ** beta
    Like proxyless nas, loss = E[ce] + E[regularizer] + E[latency_loss]
    """
    head_loss = super(ClassificationModel, self).head_loss_(x, y)
    if hasattr(self, 'loss_func'):
      # TODO(ZhouJ) This may fail in python2
      self.loss = self.loss_func(head_loss, self.blk_loss)
    else:
      # default
      self.loss = head_loss + 0.1 * self.blk_loss
    return self.loss, head_loss

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

    # tbs forward
    if tbs_input is None:
      tbs_input = {}
    for i, b in enumerate(self.tbs_blocks):
      if i == 0:
        x, self.blk_loss = b(x, **tbs_input)
      else:
        x, b_l = b(x, **tbs_input)
        self.blk_loss += b_l
    # head forward
    if head_input is None:
      x = self.head(x)
    else:
      x = self.head(x, head_input)
    l, ce = self.loss_(x, y, mode=mode)
  
    pred = torch.argmax(x, dim=1)
    acc = torch.sum(pred == y).float() / self.batch_size
    return l, ce, acc
