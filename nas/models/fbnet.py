import torch
import torch.nn as nn

from .classify_model import ClassificationModel
from ..blocks.fbnet_custom_blks import FBNetCustomBlock, FBNetCustomBlock_v1
from ..blocks.fbnet_blks import FBNetPaperBlock
from ..head.classify_head import ClassificationHead

class FBNetCustom(ClassificationModel):
  """[FBNet](https://arxiv.org/pdf/1812.03443.pdf)
  """
  def __init__(self, num_classes, alpha=0.2, beta=0.6):
    """
    Parameters
    ----------
    num_classes : int
      number of classes for classification
    

    NOTE
      - first conv : 64
      - output dim : 192
    """
    in_channels = 64
    base = nn.Conv2d(3, in_channels, 3, 1, padding=1)
    tbs_list = []
    layer = [3, 4, 6, 3]
    channels = [122, 184, 352, 1024]
    out_channels = channels[0]

    layer_idx = 0
    for i, l in enumerate(layer):
      stride = 2
      for _ in range(l):
        out_channels = channels[i]

        name = "layer_%d" % (layer_idx)
        layer_idx += 1
        tbs_list.append(FBNetCustomBlock(in_channels=in_channels,
                                         out_channels=out_channels,
                                         stride=stride,
                                         name=name))
        in_channels = out_channels
        stride = 1
    
    head = ClassificationHead(in_channels, 192, num_classes=num_classes)
    super(FBNetCustom, self).__init__(base=base,
                                      tbs_blocks=tbs_list,
                                      head=head)
    
    self.register_loss_func(lambda x, y: x + alpha * y.pow(beta))

class FBNetCustom_v1(ClassificationModel):
  """[FBNet](https://arxiv.org/pdf/1812.03443.pdf)
  """
  def __init__(self, num_classes, alpha=0.2, beta=0.6):
    """
    Parameters
    ----------
    num_classes : int
      number of classes for classification
    

    NOTE
      - first conv : 64
      - output dim : 192
    """
    in_channels = 64
    base = nn.Conv2d(3, in_channels, 3, 1, padding=1)
    tbs_list = []
    layer = [2, 2, 2, 2]
    channels = [128, 256, 512, 1024]
    out_channels = channels[0]

    layer_idx = 0
    for i, l in enumerate(layer):
      stride = 2
      for _ in range(l):
        out_channels = channels[i]

        name = "layer_%d" % (layer_idx)
        layer_idx += 1
        tbs_list.append(FBNetCustomBlock_v1(in_channels=in_channels,
                                         out_channels=out_channels,
                                         stride=stride,
                                         name=name))
        in_channels = out_channels
        stride = 1
    
    head = ClassificationHead(in_channels, 192, num_classes=num_classes)
    super(FBNetCustom_v1, self).__init__(base=base,
                                      tbs_blocks=tbs_list,
                                      head=head)
    
    self.register_loss_func(lambda x, y: x + alpha * y.pow(beta))

class FBNetCustom_v1_224(ClassificationModel):
  """[FBNet](https://arxiv.org/pdf/1812.03443.pdf)
  """
  def __init__(self, num_classes, alpha=0.2, beta=0.6):
    """
    Parameters
    ----------
    num_classes : int
      number of classes for classification
    

    NOTE
      - first conv : 64
      - output dim : 192
    """
    in_channels = 64
    base = nn.Conv2d(3, in_channels, 3, 2, padding=1)
    tbs_list = []
    layer = [2, 2, 2, 2]
    channels = [128, 256, 512, 1024]
    out_channels = channels[0]

    layer_idx = 0
    for i, l in enumerate(layer):
      stride = 2
      for _ in range(l):
        out_channels = channels[i]

        name = "layer_%d" % (layer_idx)
        layer_idx += 1
        tbs_list.append(FBNetCustomBlock_v1(in_channels=in_channels,
                                         out_channels=out_channels,
                                         stride=stride,
                                         name=name))
        in_channels = out_channels
        stride = 1
    
    head = ClassificationHead(in_channels, 192, num_classes=num_classes)
    super(FBNetCustom_v1_224, self).__init__(base=base,
                                      tbs_blocks=tbs_list,
                                      head=head)
    
    self.register_loss_func(lambda x, y: x + alpha * y.pow(beta))

class FBNet(ClassificationModel):
  """[FBNet](https://arxiv.org/pdf/1812.03443.pdf)
  """
  def __init__(self, num_classes, alpha=0.2, beta=0.6):
    """
    Parameters
    ----------
    num_classes : int
      number of classes for classification
    

    NOTE
      - first conv : 64
      - output dim : 192
    """
    
    tbs_list = []
    _f = [16, 24, 32, 
          64, 112, 184, 352]
    _n = [1, 4, 4,
          4, 4, 4, 1]
    _s = [1, 2, 2,
          2, 1, 2, 1]
    in_channels = 16
    base = nn.Conv2d(3, in_channels, 3, 1, padding=1)
    out_channels = channels[0]

    layer_idx = 0
    for i, l in enumerate(_f):
      stride = _s[i]
      for _ in range(l):
        out_channels = channels[i]

        name = "layer_%d" % (layer_idx)
        layer_idx += 1
        tbs_list.append(FBNetPaperBlock(in_channels=in_channels,
                                        out_channels=out_channels,
                                        stride=stride,
                                        name=name))
        in_channels = out_channels
        stride = 1
    
    head = ClassificationHead(in_channels, 1984, num_classes=num_classes)
    super(FBNet, self).__init__(base=base,
                                tbs_blocks=tbs_list,
                                head=head)
    
    self.register_loss_func(lambda x, y: x + alpha * y.pow(beta))
