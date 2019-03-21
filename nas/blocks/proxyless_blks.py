
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

    self.build_from_config(self.default_config())
    self.init_arch_params()

  def default_config(self):
    if self._default_cfg is None:
      cfg = []
      for kernel_size in [3, 5, 7]:
        for expansion in [1, 3]:
          kwargs = {'in_channels' : self.in_channels,
                    'out_channels' : self.out_channels,
                    'kernel_size' : kernel_size,
                    'stride' : self.stride,
                    'expand_ratio' : expansion}
          cfg.append(['MBInvertedConvLayer', kwargs])
      self._default_cfg = cfg
    return self._default_cfg

class ProxylessBlock_v1(SampleBlock):
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
    super(ProxylessBlock_v1, self).__init__(**kwargs)
    self._default_cfg = None

    self.build_from_config(self.default_config())
    self.init_arch_params()

  def default_config(self):
    if self._default_cfg is None:
      cfg = []
      for b_idx, g in enumerate([1, 2, 1, 2, 1]):
        if b_idx < 2:
          kwargs = {'inplanes' : self.in_channels,
                    'planes' : self.out_channels,
                    'stride' : self.stride,
                    'groups' : g}
          cfg.append(['ResNetBasicBlock', kwargs])
        elif b_idx >=2 and b_idx < 4:
          kwargs = {'inplanes' : self.in_channels,
                    'planes' : self.out_channels,
                    'stride' : self.stride,
                    'groups' : g}
          cfg.append(['ResNetBottleneck', kwargs])
        else:
          kwargs = {'inplanes' : self.in_channels,
                    'planes' : self.out_channels,
                    'stride' : self.stride}
          cfg.append(['SEBasicBlock', kwargs])
      if (self.in_channels == self.out_channels) and (self.stride == 1):
        cfg.append(['Identity', {}])
      self._default_cfg = cfg
    return self._default_cfg
