import torch

def acc_func(o, t, batch_size):
  pred = torch.argmax(o, dim=1)
  # TODO(ZhouJ) ugly way
  if t.device != pred.device:
    t = t.to(pred.device)
  acc = torch.sum(pred == t).float() / batch_size
  return acc
