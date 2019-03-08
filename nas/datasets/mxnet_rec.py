
import mxnet as mx
import torch

from .fix_len_iterator import FixLenIter

def mxndarray2thtensor(x):
  x = x.asnumpy()
  return torch.tensor(x)

class MXIterWrapper(object):
  """MXNet ImageRecordIter wrapper.
  
  """
  def __init__(self, mx_ds,
               train_len,
               valid_len):
    """

    Parameters
    ----------
    mx_ds : mxnet dataset

    train_len : int
      length of train ds
    valid_len : int
      length of valid ds
    
    """
    self.mx_ds = mx_ds
    self.mx_iter = iter(mx_ds)
    self.train_len = train_len
    self.valid_len = valid_len

  def __next__(self):
    try:
      next_batch = next(self.mx_iter)
    except StopIteration:
      self.mx_ds.reset()
      self.mx_iter = iter(self.mx_ds)
      next_batch = next(self.mx_iter)
    data = next_batch.data
    data = list(map(mxndarray2thtensor, data))
    if len(data) == 1:
      data = data[0]
    label = next_batch.label
    label = list(map(mxndarray2thtensor, label))
    label = list(map(lambda x: x.to(torch.long), label))
    if len(label) == 1:
      label = label[0]
    return data, label
  
  def get_train(self):
    return FixLenIter(self, self.train_len)
  
  def get_valid(self):
    return FixLenIter(self, self.valid_len)
