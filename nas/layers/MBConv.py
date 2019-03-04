
import torch.nn as nn
from collections import OrderedDict

from .utils import get_same_padding
from . import BASICUNIT

@BASICUNIT.register_module
class MBInvertedConvLayer(nn.Module):
  """MobileNetV2 basic building block.
  """

  def __init__(self, in_channels, out_channels, 
               kernel_size=3, stride=1, expand_ratio=6):
    """
    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    stride : int
    expand_ratio : int
    """
    super(MBInvertedConvLayer, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.expand_ratio = expand_ratio

    if self.expand_ratio > 1:
      feature_dim = round(in_channels * self.expand_ratio)
      self.inverted_bottleneck = nn.Sequential(OrderedDict([
          ('conv', nn.Conv2d(in_channels, feature_dim, 1, 1, 0, bias=False)),
          ('bn', nn.BatchNorm2d(feature_dim)),
          ('relu', nn.ReLU6(inplace=True)),
      ]))
    else:
      feature_dim = in_channels
      self.inverted_bottleneck = None

    # depthwise convolution
    pad = get_same_padding(self.kernel_size)
    self.depth_conv = nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, 
                           stride, pad, groups=feature_dim, bias=False)),
        ('bn', nn.BatchNorm2d(feature_dim)),
        ('relu', nn.ReLU6(inplace=True)),
    ]))

    # pointwise linear
    self.point_linear = OrderedDict([
        ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
    ])

    self.point_linear = nn.Sequential(self.point_linear)

  def forward(self, x):
      if self.inverted_bottleneck:
          x = self.inverted_bottleneck(x)
      x = self.depth_conv(x)
      x = self.point_linear(x)
      return x
