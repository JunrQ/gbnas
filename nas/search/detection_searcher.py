import logging
import time
from mmdet.apis.train import parse_losses

from .base_searcher import BaseSearcher
from ..utils import AvgrageMeter
from .utils import acc_func

class DetectionSearcher(BaseSearcher):
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
               mmcv_parallel=False,
               imgs_per_gpu=2,
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
    super(DetectionSearcher, self).__init__(
      model=model, mod_opt_dict=mod_opt_dict,
      arch_opt_dict=arch_opt_dict, gpus=gpus, 
      logger=logger, mmcv_parallel=mmcv_parallel, **kwargs)
    self.batch_size = imgs_per_gpu * len(gpus)
    
    # Info
    self._loss_avg = AvgrageMeter('loss')
    self.avgs = []

    # ds
    self.w_ds = train_w_ds
    self.arch_ds = train_arch_ds
  
  def _step_forward(self, *args, **kwargs):
    """Perform one forward step.

    Take inputs, return loss.
    Modify some attributes.
    """
    if self.decay_temperature:
      kwargs['tbs_input'] = {'temperature' :  self.temperature}
    losses = self.mod(*args, **kwargs)

    loss, log_vars = parse_losses(losses)
    self.cur_batch_loss = loss # sum(loss.values())
    return self.cur_batch_loss
