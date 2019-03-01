from torch.optim.lr_scheduler import _LRScheduler
import torch

def weights_init(m, deepth=0, max_depth=2):
  if deepth > max_depth:
    return
  if isinstance(m, torch.nn.Conv2d):
    torch.nn.init.kaiming_uniform_(m.weight.data)
    if m.bias is not None:
      torch.nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, torch.nn.Linear):
    m.weight.data.normal_(0, 0.01)
    if m.bias is not None:
      m.bias.data.zero_()
  elif isinstance(m, torch.nn.BatchNorm2d):
    return
  elif isinstance(m, torch.nn.ReLU):
    return
  elif isinstance(m, torch.nn.Module):
    deepth += 1
    for m_ in m.modules():
      weights_init(m_, deepth)
  else:
    raise ValueError("%s is unk" % m.__class__.__name__)

class CosineDecayLR(_LRScheduler):
  def __init__(self, optimizer, T_max, alpha=1e-4,
               t_mul=2, lr_mul=0.9,
               last_epoch=-1,
               warmup_step=300,
               logger=None):
    self.T_max = T_max
    self.alpha = alpha
    self.t_mul = t_mul
    self.lr_mul = lr_mul
    self.warmup_step = warmup_step
    self.logger = logger
    self.last_restart_step = 0
    self.flag = True
    super(CosineDecayLR, self).__init__(optimizer, last_epoch)

    self.min_lrs = [b_lr * alpha for b_lr in self.base_lrs]
    self.rise_lrs = [1.0 * (b - m) / self.warmup_step 
                     for (b, m) in zip(self.base_lrs, self.min_lrs)]

  def get_lr(self):
    T_cur = self.last_epoch - self.last_restart_step
    assert T_cur >= 0
    if T_cur <= self.warmup_step and (not self.flag):
      base_lrs = [min_lr + rise_lr * T_cur
              for (base_lr, min_lr, rise_lr) in 
                zip(self.base_lrs, self.min_lrs, self.rise_lrs)]
      if T_cur == self.warmup_step:
        self.last_restart_step = self.last_epoch
        self.flag = True
    else:
      base_lrs = [self.alpha + (base_lr - self.alpha) *
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs]
    if T_cur == self.T_max:
      self.last_restart_step = self.last_epoch
      self.min_lrs = [b_lr * self.alpha for b_lr in self.base_lrs]
      self.base_lrs = [b_lr * self.lr_mul for b_lr in self.base_lrs]
      self.rise_lrs = [1.0 * (b - m) / self.warmup_step 
                     for (b, m) in zip(self.base_lrs, self.min_lrs)]
      self.T_max = int(self.T_max * self.t_mul)
      self.flag = False
    
    return base_lrs

class AvgrageMeter(object):

  def __init__(self, name=''):
    self.reset()
    self._name = name

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
  
  def __str__(self):
    return "%s: %.5f" % (self._name, self.avg)
  
  def __repr__(self):
    return self.__str__()
