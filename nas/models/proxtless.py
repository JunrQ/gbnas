

from base_model import BaseModel


class ProxylessNAS(BaseModel):
  """ProxylessNAS
  [PROXYLESSNAS](https://arxiv.org/abs/1812.00332)
  """




from ..blocks.proxyless_blks import ProxylessBlock
from ..head.classify_head import ClassificationHead



  