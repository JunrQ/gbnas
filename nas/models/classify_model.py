

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

  def loss_(self, x, y):
    """Calculate loss and return it.
    """
    self.loss = super(ClassificationModel, self).loss_(x, y)


