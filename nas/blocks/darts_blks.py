
import torch.nn as nn

from .dag_blks import DAGBlock

class DartsPaperBlock(DAGBlock):
  """[darts](https://github.com/quark0/darts)
  """
  def __init__(self, **kwargs):
    super(DartsPaperBlock, self).__init__(**kwargs)
    self._default_cfg = None

    self.build_from_config(self.default_config())
    self.init_arch_params()

  def default_config(self):
    if self._default_cfg is None:
      cfg = []

      cfg.append(['AvgPool2d', {'kernel_size' : 3,
                                'padding' : 1,
                                'stride' : self.stride}])
      cfg.append(['MaxPool2d', {'kernel_size' : 3,
                                'padding' : 1,
                                'stride' : self.stride}])
      if self.stride == 1:
        cfg.append(['Identity', {}])
      else:
        cfg.append(['FactorizedReduce', {'C_in' : self.out_channels,
                                         'C_out' : self.out_channels}])

      cfg.append(['SepConv', {'C_in' : self.out_channels,
                              'C_out' : self.out_channels,
                              'kernel_size' : 3,
                              'stride' : self.stride,
                              'padding' : 1,
                              'affine' : False}])
      cfg.append(['SepConv', {'C_in' : self.out_channels,
                              'C_out' : self.out_channels,
                              'kernel_size' : 5,
                              'stride' : self.stride,
                              'padding' : 2,
                              'affine' : False}])
      cfg.append(['SepConv', {'C_in' : self.out_channels,
                              'C_out' : self.out_channels,
                              'kernel_size' : 7,
                              'stride' : self.stride,
                              'padding' : 3,
                              'affine' : False}])
      cfg.append(['DilConv', {'C_in' : self.out_channels,
                              'C_out' : self.out_channels,
                              'kernel_size' : 3,
                              'stride' : self.stride,
                              'dilation' : 2,
                              'padding' : 2,
                              'affine' : False}])
      cfg.append(['DilConv', {'C_in' : self.out_channels,
                              'C_out' : self.out_channels,
                              'kernel_size' : 5,
                              'stride' : self.stride,
                              'dilation' : 2,
                              'padding' : 4,
                              'affine' : False}])

      self._default_cfg = cfg
    return self._default_cfg
