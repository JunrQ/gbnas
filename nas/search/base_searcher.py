import torch
import torch.optim as optim
from torch.nn.parallel import DataParallel
import logging

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
               arch_sche_cfg=None):
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
      self.mod = self.mod.cuda()
    if len(gpus) > 1:
      self.mod = DataParallel(self.mod, gpus)

    # Log info
    self.logger = logger
  
  def search(self, **kwargs):
    """Search architecture.
    """
    raise NotImplementedError()

  def step_w(self, inputs, target):
    """Perform one step of $w$ training.

    Parameters
    ----------
    inputs : list or tuple of four elemets
      e.g. (x, None, None, None)
    targets : 
      calculating loss
    """
    self.mode = 'w'
    if self.cuda:
      inputs = inputs.cuda()
      target = target.cuda()
    self.w_opt.zero_grad()
    _, loss = self._step_forward(inputs, y=target, mode='w')
    loss.backward()
    self.w_opt.step()
    if self.w_lr_scheduler:
      self.w_lr_scheduler.step()

  def step_arch(self, inputs, target):
    """Perform one step of arch param training.

    Parameters
    ----------
    inputs : list or tuple of four elemets
      e.g. (x, None, None, None)
    targets : 
      calculating loss
    """
    self.mode = 'a'
    if self.cuda:
      inputs = inputs.cuda()
      target = target.cuda()
    self.a_opt.zero_grad()
    _, loss = self._step_forward(inputs, y=target, mode='a')
    loss.backward()
    self.a_opt.step()
    if self.arch_lr_scheduler:
      self.arch_lr_scheduler.step()

  def _step_forward(self, inputs, y, mode='w'):
    """Perform one forward step.
    """
    self.cur_batch_target = y
    self.cur_batch_output, loss = self.mod(x=inputs, y=y, mode=mode)
    self.batch_size = inputs.size()[0]
    if isinstance(loss, (list, tuple)):
      self.cur_batch_loss = loss[0]
      self.cur_batch_ce = loss[1]
    if len(self.gpus) > 1:
      self.cur_batch_loss = self.cur_batch_loss.mean()
      self.cur_batch_ce = self.cur_batch_ce.mean()
    return self.cur_batch_output, self.cur_batch_loss

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
  
  def batch_end_callback(self, epoch, batch, log=False):
    """Callback.

    Parameters
    ----------
    batches : int
      current batches
    log : bool
      whether do logging
    """
    raise NotImplementedError()
  
  def log_info(self):
    raise NotImplementedError()
  
  def add_avg(self, avg):
    """Add an avg object.
    """
    self.avgs.append(avg)



