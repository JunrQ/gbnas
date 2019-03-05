
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
    """
    head_loss = super(ClassificationModel, self).head_loss_(x, y)
    if hasattr(self, 'loss_func'):
      # TODO(ZhouJ) This may fail in python2
      self.loss = self.loss_func(head_loss, self.blk_loss)
    else:
      # default
      self.loss = head_loss + 0.1 * self.blk_loss
    return (self.loss, head_loss)
