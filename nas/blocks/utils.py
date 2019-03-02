import torch
import time

def tensor2list(t):
  """Transfer torch.tensor to list.
  """
  return list(t.detach().long().cpu().numpy())

def scalar2int(t):
  """Transfer scalar to int.
  """
  return int(t.detach().long().cpu().numpy())

def measure_speed(net, 
            input,
            device='cuda',
            times=500):
  """Given blocks, input and device,
  measure speed
  """

  net.to(device)
  input = input.to(device)
  output = net(input)

  tic = time.time()
  for _ in range(times):
    output = net(input)
  toc = time.time()
  speed = 1.0 * (toc - tic) / times

  return speed