import numpy as np

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])

def  get_cifar10_v1(root='/home/zhouchangqing/nas/',
                    train_portion=0.7,
                    batch_size=128):
  """Get cifar 10 dataset.

  TODO(ZhouJ)

  Paramters
  ---------
  root : str
  train_portion : float
    ratio of train
  batch_size : int
  """
  train_data = dset.CIFAR10(root=root, train=True, 
                  download=False, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size,
    shuffle=True, pin_memory=True, num_workers=16)

  val_queue = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size,
    pin_memory=True, num_workers=8)

  return train_queue, val_queue

def get_cifar10(root, train_portion, batch_size):
  # TODO
  pass
