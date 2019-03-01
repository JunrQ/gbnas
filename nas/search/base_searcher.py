
import torch.optim as optim
import logging

from ..models.base_model import BaseModel

class BaseSearcher(object):
  """Base class for searching network.
  """

  def __init__(self, model,
               mod_opt_dict,
               arch_opt_dict,
               gpus,
               logger=logging):
    """
    Parameters
    ----------


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

    # Log info
    self.logger = logger

  def _step_train(self, inputs, target):
    """Perform one step of $w$ training.

    Parameters
    ----------
    inputs : list or tuple of four elemets
      e.g. (x, None, None, None)
    targets : 
      calculating loss
    """
    self.w_opt.zero_grad()
    output = self.mod(*input)
    loss = self.mod.loss(output, target)
    loss.backward()
    self.w_opt.step()

  def _step_search(self, inputs, target):
    """Perform one step of arch param training.

    Parameters
    ----------
    inputs : list or tuple of four elemets
      e.g. (x, None, None, None)
    targets : 
      calculating loss
    """
    self.a_opt.zero_grad()
    output = self.mod(*input)
    loss = self.mod.loss(output, target)
    loss.backward()
    self.a_opt.step()

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

