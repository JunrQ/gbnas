import torch
import torch.nn as nn

from ..models.detection_model import DetectionModel
from ..head.detection_head import DetectionHead
from ..blocks.fbnet_custom_blks import FBNetCustomBlock

class FBNetCustomFasterRCNN(DetectionModel):
  """Test class for detection.
  """
  def __init__(self, cfg, train_cfg, test_cfg, channels):

    in_channels = 64
    base = nn.Sequential(
      nn.Conv2d(3, in_channels, 3, padding=1, bias=False),
      nn.BatchNorm2d(in_channels))
    tbs_list = []
    layer = [2, 2, 2, 2]
    output_indices = (1, 3, 5, 7)
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

    head = DetectionHead(cfg, train_cfg, test_cfg)
    super(FBNetCustomFasterRCNN, self).__init__(base=base,
                                      tbs_blocks=tbs_list,
                                      head=head, output_indices=output_indices)
