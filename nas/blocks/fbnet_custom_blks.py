
import torch.nn as nn

from .weigthed_sum_blks import WeightedSum

class FBNetCustomBlock(WeightedSum):
  """[FBNet](https://arxiv.org/pdf/1812.03443.pdf)
  """
  def __init__(self, **kwargs):
    """FBNet blocks.
    
    Parameters
    ----------
    in_channels : int
    out_channels : int
    name : str
    device : str or torch.device
    """
    super(FBNetCustomBlock, self).__init__(**kwargs)
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

class FBNetCustomBlock_v1(WeightedSum):
  """[FBNet](https://arxiv.org/pdf/1812.03443.pdf)
  """
  def __init__(self, **kwargs):
    """FBNet blocks.
    
    Parameters
    ----------
    in_channels : int
    out_channels : int
    name : str
    device : str or torch.device
    """
    super(FBNetCustomBlock_v1, self).__init__(**kwargs)
    self._default_cfg = None

    self.build_from_config(self.default_config())
    self.init_arch_params()

  def default_config(self):
    if self._default_cfg is None:
      cfg = []
      for b_idx, g in enumerate([1, 2, 4, 1, 1]):
        if b_idx < 3:
          kwargs = {'inplanes' : self.in_channels,
                    'planes' : self.out_channels,
                    'stride' : self.stride,
                    'groups' : g}
          cfg.append(['ResNetBasicBlock', kwargs])
        elif b_idx == 3:
          kwargs = {'inplanes' : self.in_channels,
                    'planes' : self.out_channels,
                    'stride' : self.stride,
                    'groups' : g}
          cfg.append(['ResNetBottleneck', kwargs])
        else:
          kwargs = {'in_channels' : self.in_channels,
                    'out_channels' : self.out_channels,
                    'stride' : self.stride,
                    'dilation' : 1}
          cfg.append(['ShuffleV2BasicBlock', kwargs])
      if (self.in_channels == self.out_channels) and (self.stride == 1):
        cfg.append(['Identity', {}])
      self._default_cfg = cfg
    return self._default_cfg

class FBNetCustomBlock_v2(WeightedSum):
  """[FBNet](https://arxiv.org/pdf/1812.03443.pdf)
  """
  def __init__(self, **kwargs):
    """FBNet blocks.
    
    Parameters
    ----------
    in_channels : int
    out_channels : int
    name : str
    device : str or torch.device
    """
    super(FBNetCustomBlock_v1, self).__init__(**kwargs)
    self._default_cfg = None

    self.build_from_config(self.default_config())
    self.init_arch_params()

  def default_config(self):
    if self._default_cfg is None:
      cfg = []
      for b_idx, g in enumerate([1, 2, 1, 1, 1]):
        if b_idx < 2:
          kwargs = {'inplanes' : self.in_channels,
                    'planes' : self.out_channels,
                    'stride' : self.stride,
                    'groups' : g}
          cfg.append(['ResNetBasicBlock', kwargs])
        elif b_idx == 2:
          kwargs = {'inplanes' : self.in_channels,
                    'planes' : self.out_channels,
                    'stride' : self.stride,
                    'reduction' : 8}
          cfg.append(['SEBasicBlock', kwargs])
        elif b_idx == 3:
          kwargs = {'inplanes' : self.in_channels,
                    'planes' : self.out_channels,
                    'stride' : self.stride,
                    'groups' : g}
          cfg.append(['ResNetBottleneck', kwargs])
        else:
          kwargs = {'in_channels' : self.in_channels,
                    'out_channels' : self.out_channels,
                    'stride' : self.stride,
                    'dilation' : 1}
          cfg.append(['ShuffleV2BasicBlock', kwargs])
      if (self.in_channels == self.out_channels) and (self.stride == 1):
        cfg.append(['Identity', {}])
      self._default_cfg = cfg
    return self._default_cfg
