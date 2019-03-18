import torch
from .classify_model import ClassificationModel

class RLModel(ClassificationModel):
  """Base model for classification.
  """

  def __init__(self, **kwargs):
    """
    """
    super(RLModel, self).__init__(**kwargs)
    self.baseline = 0
    self.count = 0
  
  def loss_(self, x, y, log_p, mode=None):
    head_loss = super(ClassificationModel, self).head_loss_(x, y)

    if mode == 'w':
      self.loss = head_loss
    elif mode == 'a':
      loss = head_loss + 10 * self.blk_loss
      self.loss = (loss - self.baseline) * log_p
      self.count += 1
      self.baseline = self.baseline * (self.count - 1) / self.count + \
                      self.loss / self.count
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
        x, self.blk_loss, log_p = b(x, **tbs_input)
      else:
        x, b_l, tmp_log_p = b(x, **tbs_input)
        self.blk_loss += b_l
        log_p += tmp_log_p
    # head forward
    if head_input is None:
      x = self.head(x)
    else:
      x = self.head(x, head_input)
    l, ce = self.loss_(x, y, log_p, mode=mode)
  
    pred = torch.argmax(x, dim=1)
    acc = torch.sum(pred == y).float() / self.batch_size
    return l, ce, acc
