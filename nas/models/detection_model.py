import torch
import torch.nn as nn

from .base_model import BaseModel

class DetectionModel(BaseModel):
  """Base model for detection.
  """

  def __init__(self, base,
               tbs_blocks,
               head,
               output_indices=None):
    """
    """
    super(DetectionModel, self).__init__(base=base,
          tbs_blocks=tbs_blocks, head=head,
          output_indices=output_indices)

  def forward(self, img,
              img_meta,
              gt_bboxes,
              gt_bboxes_ignore,
              gt_labels,
              gt_masks=None,
              proposals=None,
              mode='w'):
    """Forward

    NOTE Inputs should be consistent with
    your dataset. 

    Parameters
    ----------
    img : torch.tensor
      input
    base_input
      base extra input
    tbs_input
      tbs part extra input
    head_input
      head extra input
    """
    self.batch_size = img.size()[0]
    # base forward
    x = self.base(img)
    # tbs forward
    outs = []
    for i, b in enumerate(self.tbs_blocks):
      if i == 0:
        x, self.blk_loss = b(x)
      else:
        x, b_l = b(x)
        self.blk_loss += b_l
      if i in self.output_indices:
        outs.append(x)
    if len(outs) == 0:
      outs.append(x)

    # head forward
    loss = self.head(outs, img_meta,
                      gt_bboxes,
                      gt_bboxes_ignore,
                      gt_labels,
                      gt_masks=gt_masks,
                      proposals=proposals)
    return loss
