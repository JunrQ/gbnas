

from base_model import BaseModel




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


  def ce(self):
    """Get cross entropy.
    """
    return self.head.ce_loss
  

  def acc(self):
    """Get accuracy.
    """
    return self.head.acc

  def loss(self, x, y):
    """Calculate loss and return it.
    """
    self.total_loss = super(ClassificationModel, self).loss(x, y)


