import torch
import torch.optim as optim
from torch.nn.parallel import DataParallel
import logging
import time
import os
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
               mmcv_parallel=False,
               init_temperature=1.0,
               decay_temperature_step=0,
               decay_temperature_ratio=0.9,
               decay_temperature_every_epoch=True,
               save_result_path='./result/',
               save_arch_params_frequence=5000,
               no_temperature=False):
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

    if no_temperature:
      decay_temperature_step = -1
      decay_temperature_every_epoch = False

    # Log info
    self.logger = logger
    self.temperature = init_temperature
    self.decay_temperature = False
    self.decay_temperature_ratio = decay_temperature_ratio
    if decay_temperature_step > 0:
      assert no_temperature or not decay_temperature_every_epoch
      self.decay_temperature_step = decay_temperature_step
      self.decay_temperature = True
    
    self._batch_end_cb_func = []
    self.add_batch_end_callback(lambda x, y: self._update_avg(x, y))
    self.add_batch_end_callback(lambda x, y: self._test_log(x, y))
    self.save_arch_params_frequence = save_arch_params_frequence
    self.add_batch_end_callback(lambda x, y:self._save_result_cb(x, y))

    if self.decay_temperature:
      self.add_batch_end_callback(lambda x, y: self._update_temp(x, y))
    
    self._epoch_end_cb_func = []
    if decay_temperature_every_epoch:
      self.decay_temperature = True # not use
      assert no_temperature or decay_temperature_step <= 0
      self.add_epoch_end_callback(lambda x: self._update_temp(x))
    self.add_epoch_end_callback(lambda x: self._save_result_cb(x))
    
    self.save_result_path = save_result_path
    if not os.path.exists(save_result_path):
      raise ValueError("Directory for save result don't exist %s" % save_result_path)
  
  def search(self, **kwargs):
    """Search architecture.
    """
    raise NotImplementedError()

  def step_w(self, *inputs, **kwargs):
    """Perform one step of $w$ training.
    """
    self.mode = 'w'
    self.w_opt.zero_grad()
    kwargs['mode'] = 'w'
    loss = self._step_forward(*inputs, **kwargs)
    loss.backward()
    self.w_opt.step()
    if self.w_lr_scheduler:
      self.w_lr_scheduler.step()

  def step_arch(self, *inputs, **kwargs):
    """Perform one step of arch param training.
    """
    self.mode = 'a'
    self.a_opt.zero_grad()
    kwargs['mode'] = 'a'
    loss = self._step_forward(*inputs, **kwargs)
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
      a.reset()
    self.logger.info(msg)
    # map is delayed
    # map(lambda avg: avg.reset(), self.avgs)
    return msg
  
  def _update_avg(self, epoch, batch):
    for avg in self.avgs:
      if avg.name == 'acc':
        avg.update(self.cur_batch_acc)
      elif avg.name == 'ce':
        avg.update(self.cur_batch_ce)
      else:
        value = avg.cal(self).detach().numpy()
        avg.update(value)
    self._loss_avg.update(self.cur_batch_loss)
  
  def _test_log(self, epoch, batch):
    if (batch > 0) and (batch % self.log_frequence == 0):
      self.toc = time.time()
      speed = 1.0 * (self.batch_size * self.log_frequence) / (self.toc - self.tic)
      self.log_info(epoch, batch, speed=speed)
      self.tic = time.time()
  
  def _update_temp(self, epoch, batch=None):
    """
    Parameters
    ----------
    batch : None or int
      None means epoch callback
    """
    if (batch is None) or ((batch > 0) and (batch % self.decay_temperature_step == 0)):
      self.temperature *= self.decay_temperature_ratio
      msg = 'Epoch[%d] ' % epoch
      if not batch is None:
        msg += 'Batch[%d] ' % batch
      msg += "Decay temperature to %.6f" % self.temperature
      self.logger.info(msg)
  
  def _save_result_cb(self, epoch, batch=None):
    if (batch is None ) or ((batch > 0) and (batch % self.save_arch_params_frequence == 0)):
      if not batch is None:
        save_path = "%s/epoch_%d_batch_%d_arch_params.txt" % \
                        (self.save_result_path, epoch, batch)
      else:
        save_path = "%s/epoch_%d_end_arch_params.txt" % \
                        (self.save_result_path, epoch)
      self.save_arch_params(save_path)
      self.logger.info("Save architecture paramterse to %s" % save_path)
  
  def add_batch_end_callback(self, func):
    self._batch_end_cb_func.append(func)
  
  def add_epoch_end_callback(self, func):
    self._epoch_end_cb_func.append(func)
  
  def batch_end_callback(self, epoch, batch):
    """Callback.

    Parameters
    ----------
    epoch : int
      current epoch
    batch : int 
      current batch
    """
    for func in self._batch_end_cb_func:
      func(epoch, batch)
  
  def epoch_end_callback(self, epoch):
    for func in self._epoch_end_cb_func:
      func(epoch)

  def add_avg(self, avg):
    """Add an avg object.
    """
    self.avgs.append(avg)

  def search(self, **kwargs):
    """Override this method if you need a different
    search procedure.

    Parameters
    ----------
    epoch : int
      number of epochs, default is 100
    start_w_epoch : int
      train w for start_w_epoch epochs before training
      architecture parameters, default is 5
    log_frequence : int
      log every log_frequence batches, defaulit is 50
    """

    num_epoch = kwargs.get('epoch', 100)
    start_w_epoch = kwargs.get('start_w_epoch', 5)
    self.log_frequence = kwargs.get('log_frequence', 50)

    assert start_w_epoch >= 1, "Start to train w first"

    for epoch in range(start_w_epoch):
      self.tic = time.time()
      self.logger.info("Start to train w for epoch %d" % epoch)
      for step, (input, target) in enumerate(self.w_ds):
        self.step_w(input, target)
        self.batch_end_callback(epoch, step)
      self.epoch_end_callback(epoch)

    for epoch in range(num_epoch):
      self.tic = time.time()
      self.logger.info("Start to train arch for epoch %d" % (epoch+start_w_epoch))
      for step, (input, target) in enumerate(self.arch_ds):
        self.step_arch(input, target)
        self.batch_end_callback(epoch+start_w_epoch, step)
        
      self.tic = time.time()
      self.logger.info("Start to train w for epoch %d" % (epoch+start_w_epoch))
      for step, (input, target) in enumerate(self.w_ds):
        self.step_w(input, target)
        self.batch_end_callback(epoch+start_w_epoch, step)
      self.epoch_end_callback(epoch+start_w_epoch)
