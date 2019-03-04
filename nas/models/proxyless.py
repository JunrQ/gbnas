import torch
import torch.nn as nn

from .classify_model import ClassificationModel
from ..blocks.proxyless_blks import ProxylessBlock
from ..head.classify_head import ClassificationHead

class ProxylessNAS(ClassificationModel):
  """ProxylessNAS
  [PROXYLESSNAS](https://arxiv.org/abs/1812.00332)
  """

  def __init__(self, num_classes):
    """
    Parameters
    ----------
    num_classes : int
      number of classes for classification
    """
    in_channels = 64
    base = nn.Conv2d(3, in_channels, 3, 1, padding=1)
    tbs_list = []
    layer = [3, 3, 3]
    channels = [112, 184, 352]
    out_channels = channels[0]

    layer_idx = 0
    for i, l in enumerate(layer):
      stride = 2
      for _ in range(l):
        out_channels = channels[i]

        name = "layer_%d" % (layer_idx)
        layer_idx += 1
        tbs_list.append(ProxylessBlock(in_channels=in_channels,
                                      out_channels=out_channels,
                                      stride=stride,
                                      name=name))
        in_channels = out_channels
        stride = 1
    
    head = ClassificationHead(in_channels, 192, num_classes=num_classes)
    super(ProxylessNAS, self).__init__(base=base,
                                       tbs_blocks=tbs_list,
                                       head=head)
  

  def loss_(self, x, y, mode='w'):
    """
    Parameters
    ----------
    x : tensor
      input
    y : tensor
      target
    mode : str
      'w' for training model parameters
      'a' for training architecture parameters
      default is 'w'

    TODO(ZhouJ) Loss in original paper
    """
    if mode is None:
      mode = 'w'
    head_loss = super(ClassificationModel, self).head_loss_(x, y)
    if mode == 'w':
      self.loss = head_loss + 0 * self.blk_loss
    elif mode == 'a':
      self.loss = head_loss + 0.1 * blkself.blk_loss_loss
    else:
      raise ValueError("Not supported mode: %s provided" % mode)
    return self.loss
