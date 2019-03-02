

from classify_model import ClassificationModel


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
    layer = [3, 4, 6, 3]
    channels = [112, 184, 352, 1024]
    out_channels = channels[0]
    for i, l in enumerate(layer):
      stride = 2

      for j in range(l):
        out_channels = channels[i]
        tbs_list.append(ProxylessBlock(in_channels=in_channels,
                                      out_channels=out_channels,
                                      stride=stride))
        in_channels = out_channels
        stride = 1
    
    head = ClassificationHead(in_channels, 192)

    super(ProxylessNAS, self).__init__(base=base,
                                       tbs_blocks=tbs_list,
                                       head=head)
    


