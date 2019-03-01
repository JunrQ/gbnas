
import torch.optim as optim


class BaseSearcher(object):
  """Base class for searching network.
  """

  def __init__(self, model,
               mod_opt_dict,
               arch_opt_dict):
    """
    Parameters
    ----------


    """

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

  
  def _step_train(self, input, target):
    """Perform one step of $w$ training.
    """
    self.w_opt.zero_grad()
    output = self.mod(input)
    loss = self.mod.loss(output, target)
    loss.backward()
    self.w_opt.step()
  


  def _step_search(self, input, target):
    """Perform one step of arch param training.
    """
    self.a_opt.zero_grad()
    output = self.mod(input)
    loss = self.mod.loss(output, target)
    loss.backward()
    self.a_opt.step()


  