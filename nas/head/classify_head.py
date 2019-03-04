import torch
import torch.nn as nn

from .base_head import BaseHead

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
                                    nn.Linear(in_channels, feature_dim),
                                    nn.Linear(feature_dim, num_classes))
    self._ce = nn.CrossEntropyLoss().to(device)
  
  def forward(self, x):
    self.batch_size = x.size()[0]
    x = self.global_pooling(x)
    x = self.classifier(x.view(x.size(0), -1))
    self.logits = x
    return x
  
  def loss_(self, x, target):
    self.ce_loss = self._ce(x, target)
    return self.ce_loss
