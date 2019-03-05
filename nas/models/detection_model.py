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

  def forward(self, img,
              img_meta,
              gt_bboxes,
              gt_bboxes_ignore,
              gt_labels,
              gt_masks=None,
              proposals=None,
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
    self.batch_size = img.size()[0]
    print('img', img.size())
    # base forward
    x = self.base(img)
    # tbs forward
    assert tbs_input is None, 'Not supported for now'
    for i, b in enumerate(self.tbs_blocks):
      if i == 0:
        x, self.blk_loss = b(x)
      else:
        x, b_l = b(x)
        self.blk_loss += b_l

    # head forward
    loss = self.head(x, img_meta,
                        gt_bboxes,
                        gt_bboxes_ignore,
                        gt_labels,
                        gt_masks=gt_masks,
                        proposals=proposals)
    return loss
