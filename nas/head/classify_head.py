import torch.nn as nn

from base_head import BaseHead

class ClassificationHead(BaseHead):
  """Head for classification.
  """

  def __init__(self, in_channels,
               feature_dim,
               num_classes,
               device='cuda'):

    super(ClassificationHead, self).__init__()

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Sequential(nn.BatchNorm1d(in_channels),
                                    nn.Linear(in_channels, feature_dim)
                                    nn.Linear(feature_dim, num_classes))
    self._criterion = nn.CrossEntropyLoss().to(device)
  
  def forward(self, x):
    x = self.global_pooling(x)
    x = self.classifier(x.view(x.size(0), -1))
  
  def loss(self, x, target):
    return self._criterion(x, target)








