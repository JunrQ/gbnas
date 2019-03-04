import torch

def acc_func(o, t, batch_size):
  pred = torch.argmax(o, dim=1)
  acc = torch.sum(pred == t).float() / batch_size
  return acc