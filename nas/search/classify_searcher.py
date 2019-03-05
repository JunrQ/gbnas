import logging
import time

from .base_searcher import BaseSearcher
from ..utils import AvgrageMeter
from .utils import acc_func

class ClassificationSearcher(BaseSearcher):
  """Search class for classification.
  """

  def __init__(self,
               model,
               gpus,
               train_w_ds,
               train_arch_ds,
               mod_opt_dict,
               arch_opt_dict,
               logger=logging,
               **kwargs):
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
    train_w_ds : dataset
      dataset for training w
    train_arch_ds : dataset
      dataset for traing architecture parameters
    logger : logger
    w_lr_scheduler : `subclass` of _LRScheduler
      default is CosineDecayLR
    w_sche_cfg : dict
      parameters for w_lr_scheduler
    arch_lr_scheduler : 
      default is None
    arch_sche_cfg : dict
      parameters for arch_lr_scheduler
    """
    super(ClassificationSearcher, self).__init__(
      model=model, mod_opt_dict=mod_opt_dict,
      arch_opt_dict=arch_opt_dict, gpus=gpus, 
      logger=logger, **kwargs)
    
    # Info
    self._acc_avg = AvgrageMeter('acc')
    self._acc_avg.register_func(lambda obj : 
            acc_func(getattr(obj, 'cur_batch_output'), 
                     getattr(obj, 'cur_batch_target'),
                     getattr(obj, 'batch_size')))
    self._ce_avg = AvgrageMeter('ce')
    self._ce_avg.register_func(lambda obj:
            getattr(obj, 'cur_batch_ce'))

    self._loss_avg = AvgrageMeter('loss')
    self.avgs = [self._acc_avg, self._ce_avg]

    # ds
    self.w_ds = train_w_ds
    self.arch_ds = train_arch_ds
  
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
    for avg in self.avgs:
      value = avg.cal(self)
      avg.update(value)
    self._loss_avg.update(self.cur_batch_loss)
    
    if (batch > 0) and (batch % self.log_frequence == 0):
      self.toc = time.time()
      speed = 1.0 * (self.batch_size * self.log_frequence) / (self.toc - self.tic)
      self.log_info(epoch, batch, speed=speed)
      self.tic = time.time()

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
