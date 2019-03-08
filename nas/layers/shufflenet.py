# https://github.com/miaow1988/ShuffleNet_V2_pytorch_caffe/blob/master

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normal_blks import ChannelShuffle
from . import BASICUNIT

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False),
        nn.BatchNorm2d(out_channels),
    )


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, True)


def conv_relu(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, True),
        nn.ReLU(),
    )

def conv_prelu(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, True),
        nn.PReLU(),
    )

@BASICUNIT.register_module
class ShuffleV2BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation):
        super(ShuffleV2BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.stride = stride
        channels = out_channels//2
        if stride == 1:
            assert in_channels == out_channels
            self.conv = nn.Sequential(
                conv_bn_relu(channels, channels, 1),
                conv_bn(channels, channels, 3, stride=stride, 
                    dilation=dilation, padding=dilation, groups=channels),
                conv_bn_relu(channels, channels, 1),
            )
        else:
            self.conv = nn.Sequential(
                conv_bn_relu(in_channels, channels, 1),
                conv_bn(channels, channels, 3, stride=stride, 
                    dilation=dilation, padding=dilation, groups=channels),
                conv_bn_relu(channels, channels, 1),
            )
            self.conv0 = nn.Sequential(
                conv_bn(in_channels, in_channels, 3, stride=stride, 
                    dilation=dilation, padding=dilation, groups=in_channels),
                conv_bn_relu(in_channels, channels, 1),
            )
        self.shuffle = ChannelShuffle(2)

    def forward(self, x):
        if self.stride == 1:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            x = torch.cat((x1, self.conv(x2)), 1)
        else:
            x = torch.cat((self.conv0(x), self.conv(x)), 1)
        return self.shuffle(x)
