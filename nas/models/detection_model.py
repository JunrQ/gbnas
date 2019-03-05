import torch
import torch.nn as nn

from .base_model import BaseModel

class DetectionModel(BaseModel):
  """Base model for detection.
  """

  def __init__(self, base,
               tbs_blocks,
               head):
    """
    """
    super(DetectionModel, self).__init__(base=base,
          tbs_blocks=tbs_blocks, head=head)

  def loss_(self, mode=None):
    """Calculate loss and return it.
    """
    


    




