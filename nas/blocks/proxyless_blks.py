
import torch.nn as nn

from .sample_blks import SampleBlock

class ProxylessBlock(SampleBlock):
  """ProxylessNAS
  [PROXYLESSNAS](https://arxiv.org/abs/1812.00332)
  """

  def __init__(self, **kwargs):
    """
    Parameters
    ----------
    in_channels : int
    out_channels : int
    name : str
    device : str or torch.device
    """
    super(ProxylessBlock, self).__init__(**kwargs)
    self._default_cfg = None

    self.build_from_config(self.default_config)
    self.init_arch_params()
  
  @property
  def default_config(self):
    if self._default_cfg is None:
      cfg = []
      for kernel_size in [3, 5, 7]:
        for expansion in [3, 6]:
          kwargs = {'in_channels' : self.in_channels,
                    'out_channels:' : self.out_channels,
                    'kernel_size' : kernel_size,
                    'stride' : self.stride,
                    'expand_ratio' : expansion}
          cfg.append(['MBInvertedConvLayer', kwargs])
      self._default_cfg = cfg
    return self._default_cfg 


