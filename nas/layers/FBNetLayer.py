import torch
import torch.nn as nn

from . import BASICUNIT

@BASICUNIT.register_module
class FBNetBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride,
              expansion, groups, bn=False):
    super(FBNetBlock, self).__init__()
    assert not bn, "not support bn for now"
    bias_flag = not bn
    if groups == 1:
      self.op = nn.Sequential(
        nn.Conv2d(in_channels, in_channels*expansion, 1, stride=1, padding=0,
                  groups=groups, bias=bias_flag),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels*expansion, in_channels*expansion, 3, stride=stride, 
                  padding=1, groups=in_channels*expansion, bias=bias_flag),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels*expansion, out_channels, 1, stride=1, padding=0, 
                  groups=groups, bias=bias_flag)
      )
    else:
      self.op = nn.Sequential(
        nn.Conv2d(in_channels, in_channels*expansion, 1, stride=1, padding=0,
                  groups=groups, bias=bias_flag),
        nn.ReLU(inplace=False),
        ChannelShuffle(groups),
        nn.Conv2d(in_channels*expansion, in_channels*expansion, 3, stride=stride, 
                  padding=1, groups=in_channels*expansion, bias=bias_flag),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels*expansion, out_channels, 1, stride=1, padding=0, 
                  groups=groups, bias=bias_flag),
        ChannelShuffle(groups)
      )
    res_flag = ((in_channels == out_channels) and (stride == 1))
    self.res_flag = res_flag
    # if not res_flag:
    #   if stride == 2:
    #     self.trans = nn.Conv2d(in_channels, out_channels, 3, stride=2, 
    #                           padding=1)
    #   elif stride == 1:
    #     self.trans = nn.Conv2d(in_channels, out_channels, 1, stride=1, 
    #                           padding=0)
    #   else:
    #     raise ValueError("Wrong stride %d provided" % stride)

  def forward(self, x):
    if self.res_flag:
      return self.op(x) + x
    else:
      return self.op(x) # + self.trans(x)
