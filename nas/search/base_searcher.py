
import torch.optim as optim
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
               arch_lr_scheduler=None):
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
    self.mod = model

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
    self.w_lr_scheduler =  None if w_lr_scheduler is None else w_lr_scheduler
    self.arch_lr_scheduler =  None if arch_lr_scheduler is \
                                   else arch_lr_scheduler


    



    # Log info
    self.logger = logger
  
  def search(self):
    """Search architecture.
    """
    raise NotImplementedError()

  def step_train(self, inputs, target):
    """Perform one step of $w$ training.

    Parameters
    ----------
    inputs : list or tuple of four elemets
      e.g. (x, None, None, None)
    targets : 
      calculating loss
    """
    self.w_opt.zero_grad()
    self._step(inputs, target)
    self.w_opt.step()
    if self.w_lr_scheduler:
      self.w_lr_scheduler.step()

  def step_search(self, inputs, target):
    """Perform one step of arch param training.

    Parameters
    ----------
    inputs : list or tuple of four elemets
      e.g. (x, None, None, None)
    targets : 
      calculating loss
    """
    self.a_opt.zero_grad()
    self._step(inputs, target)
    self.a_opt.step()
    if self.arch_lr_scheduler:
      self.arch_lr_scheduler.step()

  def _step(self, inputs, target):
    """Perform one step.
    """
    output = self.mod(*input)
    loss = self.mod.loss(output, target)
    loss.backward()

  def save_arch_params(self, save_path):
    """Save architecture params.
    """
    res = []
    with open(save_path, 'w') as f:
      for t in self.mod.arch_params:
        t_list = list(t.detach().cpu().numpy())
        res.append(t_list)
        s = ' '.join([str(tmp) for tmp in t_list])
        f.write(s + '\n')
    return res

