import torch
import torch.optim as optim
from torch.nn.parallel import DataParallel
import logging
import time
from mmcv.parallel import MMDataParallel

from ..models.base_model import BaseModel
from ..utils import CosineDecayLR, AvgrageMeter


class BaseSearcher(object):
  """Base class for searching network.
  """

  def __init__(self, model,
               mod_opt_dict,
               arch_opt_dict,
               gpus,
               logger=logging,
               w_lr_scheduler=CosineDecayLR,
               w_sche_cfg={'T_max':400},
               arch_lr_scheduler=None,
               arch_sche_cfg=None,
               mmcv_parallel=False):
    """
    Parameters
    ----------
    model : obj::BaseModel
      model for forward and backward
    mod_opt_dict : dict
      model parameter optimizer settings
    arch_opt_dict : dict
      architecture parameter optimizer settings
    gpus : `list` of `int`
      devices used for training
    logger : logger

    """
    assert isinstance(model, BaseModel)
    self.mod = model.train()
    self.arch_params = self.mod.arch_params

    # Build optimizer
    assert isinstance(mod_opt_dict, dict), 'Dict required' + \
           ' for mod opt parameters'
    assert isinstance(arch_opt_dict, dict), 'Dict required' + \
           ' for arch opt parameters'
    opt_type = mod_opt_dict.pop('type')
    mod_opt_dict['params'] = self.mod.model_params
    self.w_opt = getattr(optim, opt_type)(**mod_opt_dict)
    opt_type = arch_opt_dict.pop('type')
    arch_opt_dict['params'] = self.mod.arch_params
    self.a_opt = getattr(optim, opt_type)(**arch_opt_dict)
    self.w_lr_scheduler =  None if w_lr_scheduler is None \
                           else w_lr_scheduler(self.w_opt, **w_sche_cfg)
    self.arch_lr_scheduler =  None if arch_lr_scheduler is None \
                              else arch_lr_scheduler(self.a_opt, **arch_sche_cfg)
    
    self.gpus = gpus
    self.cuda = (len(gpus) > 0)
    if self.cuda:
      # TODO(ZhouJ) If not call to(device), DataParallel will
      # failed - `Broadcast function not implemented for CPU tensors`
      # It just thinks that model sits on CPU.
      self.mod.to('cuda:' + str(self.gpus[0]))
      if mmcv_parallel:
        self.mod = MMDataParallel(self.mod, gpus) # .cuda()
      else:
        self.mod = DataParallel(self.mod, gpus)

    # Log info
    self.logger = logger
  
  def search(self, **kwargs):
    """Search architecture.
    """
    raise NotImplementedError()

  def step_w(self, *inputs, **kwargs):
    """Perform one step of $w$ training.
    """
    self.mode = 'w'
    self.w_opt.zero_grad()
    loss = self._step_forward(*inputs, **kwargs, mode='w')
    loss.backward()
    self.w_opt.step()
    if self.w_lr_scheduler:
      self.w_lr_scheduler.step()

  def step_arch(self, *inputs, **kwargs):
    """Perform one step of arch param training.
    """
    self.mode = 'a'
    self.a_opt.zero_grad()
    loss = self._step_forward(*inputs, **kwargs, mode='a')
    loss.backward()
    self.a_opt.step()
    if self.arch_lr_scheduler:
      self.arch_lr_scheduler.step()

  def save_arch_params(self, save_path):
    """Save architecture params.
    """
    res = []
    with open(save_path, 'w') as f:
      for t in self.arch_params:
        t_list = list(t.detach().cpu().numpy())
        res.append(t_list)
        s = ' '.join([str(tmp) for tmp in t_list])
        f.write(s + '\n')
    return res
  
  def log_info(self, epoch, batch, speed=None):
    msg = "Epoch[%d] Batch[%d]" % (epoch, batch)
    if speed is not None:
      msg += ' Speed: %.6f samples/sec' % speed
    msg += ' %s' % self._loss_avg
    self._loss_avg.reset()
    for a in self.avgs:
      msg += " %s" % a
    self.logger.info(msg)
    map(lambda avg: avg.reset(), self.avgs)
    return msg
  
  def batch_end_callback(self, epoch, batch):
    """Callback.

    Parameters
    ----------
    batches : int
      current batches
    log : bool
      whether do logging
    """
    for avg in self.avgs:
      value = avg.cal(self)
      avg.update(value)
    self._loss_avg.update(self.cur_batch_loss)
    
    if (batch > 0) and (batch % self.log_frequence == 0):
      self.toc = time.time()
      speed = 1.0 * (self.batch_size * self.log_frequence) / (self.toc - self.tic)
      self.log_info(epoch, batch, speed=speed)
      self.tic = time.time()

  def add_avg(self, avg):
    """Add an avg object.
    """
    self.avgs.append(avg)
