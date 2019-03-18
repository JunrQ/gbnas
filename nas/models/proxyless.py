import torch
import torch.nn as nn

from .reinforce_model import RLModel
from ..blocks.proxyless_blks import ProxylessBlock
from ..head.classify_head import ClassificationHead

class ProxylessNAS(RLModel):
  """ProxylessNAS
  [PROXYLESSNAS](https://arxiv.org/abs/1812.00332)

  NOTE 
   - has 3 stride = 2
   - default feature dim is 192
   - only 9 layers
   - only test on cifar10
  """
  def __init__(self, num_classes, alpha=0.2):
    """
    Parameters
    ----------
    num_classes : int
      number of classes for classification
    """
    in_channels = 32
    base = nn.Conv2d(3, in_channels, 3, 1, padding=1)
    tbs_list = []
    layer = [3, 3, 3]
    channels = [64, 128, 256]
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

class ProxylessNAS_face(RLModel):
  """ProxylessNAS
  [PROXYLESSNAS](https://arxiv.org/abs/1812.00332)
  """
  def __init__(self, num_classes, alpha=0.2):
    in_channels = 64
    base = nn.Conv2d(3, in_channels, 3, 1, padding=1)
    tbs_list = []
    layer = [3, 3, 3, 3]
    channels = [112, 184, 352, 512]
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
    super(ProxylessNAS_face, self).__init__(base=base,
                                       tbs_blocks=tbs_list,
                                       head=head)
