import torch.nn as nn

from .dag_model import DAGModel
from ..blocks.darts_blks import DartsPaperBlock
from ..head.classify_head import ClassificationHead


class DARTS(DAGModel):
  def __init__(self, num_classes, alpha=0.2):
    """
    NOTE
      - first conv : 64
      - output dim : 192
    """
    in_channels = 16
    base = nn.Sequential(
      nn.Conv2d(3, in_channels, 3, padding=1, bias=False),
      nn.BatchNorm2d(in_channels))

    tbs_list = []
    layer = [2, 2, 2]
    
    C = 16
    multiplier = 4
    stride = 1
    reduction_prev = False
    C_prev_prev, out_channels, in_channels = in_channels, in_channels, C

    layer_idx = 0
    for i, l in enumerate(layer):
      if i > 0:
        stride = 2
        out_channels *= 2
      for _ in range(l):

        name = "layer_%d" % (layer_idx)
        layer_idx += 1
        tbs_list.append(DartsPaperBlock(in_channels=in_channels,
                                        out_channels=out_channels,
                                        stride=stride,
                                        name=name,
                                        C_prev_prev=C_prev_prev,
                                        steps=4,
                                        multiplier=multiplier,
                                        reduction_prev=reduction_prev))

        C_prev_prev, in_channels = in_channels, multiplier*out_channels
        reduction_prev = (stride == 2)
        stride = 1
    
    head = ClassificationHead(in_channels, 192, num_classes=num_classes)
    super(DARTS, self).__init__(base=base,
                                tbs_blocks=tbs_list,
                                head=head)
    self.register_loss_func(lambda x, y: x + alpha * y)
