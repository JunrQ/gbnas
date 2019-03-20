import argparse
import time
import os
import torch

from ..models.fbnet import FBNetCustom, FBNetCustom_v1, FBNet
from ..search.classify_searcher import ClassificationSearcher
from ..datasets.cifar10 import get_cifar10_v1
from ..utils import _set_file, _logger

class Config(object):
  alpha = 0.2
  beta = 0.6
  w_lr = 0.1
  w_mom = 0.9
  w_wd = 1e-4
  t_lr = 0.01
  t_wd = 5e-4
  t_beta = (0.9, 0.999)
  model_save_path = '/home1/nas/fbnet-pytorch/fbnet/cifar10/'
  start_w_epoch = 2
  train_portion = 0.8
  init_temperature = 5.0
  decay_temperature_ratio = 0.956
  decay_temperature_step = 50
  save_frequence = 50

lr_scheduler_params = {
  'logger' : _logger,
  'T_max' : 100,
  'alpha' : 1e-4,
  'warmup_step' : 100,
  't_mul' : 1.1,
  'lr_mul' : 0.95,
}

config = Config()

parser = argparse.ArgumentParser(description="Train a model with data parallel for base net \
                                and model parallel for classify net.")
parser.add_argument('--batch-size', type=int, default=256,
                    help='training batch size of all devices.')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of training epochs.')
parser.add_argument('--log-frequence', type=int, default=400,
                    help='log frequence, default is 400')
parser.add_argument('--gpus', type=str, default='0',
                    help='gpus, default is 0')
args = parser.parse_args()

args.model_save_path = '%s/%s/' % \
            (config.model_save_path, time.strftime('%Y-%m-%d', time.localtime(time.time())))

if not os.path.exists(args.model_save_path):
  _logger.warn("{} not exists, create it".format(args.model_save_path))
  os.makedirs(args.model_save_path)
_set_file(args.model_save_path + 'log.log')


train_ds, val_ds = get_cifar10_v1(train_portion=config.train_portion,
                                  batch_size=args.batch_size)
model = FBNetCustom_v1(10, alpha=config.alpha, beta=config.beta)

# TODO(ZhouJ) put this into model or searcher
model.speed_test(torch.randn((1, 3, 32, 32)), verbose=False,
                 device='cuda:' + args.gpus[-1])

searcher = ClassificationSearcher(
              model=model,
              mod_opt_dict={'type':'SGD',
                            'lr':config.w_lr,
                            'momentum':config.w_mom,
                            'weight_decay':config.w_wd},
              arch_opt_dict={'type':'Adam',
                             'lr':config.t_lr,
                             'betas':config.t_beta,
                             'weight_decay':config.t_wd},
              logger=_logger,
              gpus=[int(x) for x in args.gpus.split(',')],
              train_w_ds=train_ds,
              train_arch_ds=val_ds,
              w_sche_cfg=lr_scheduler_params,
              init_temperature=config.init_temperature,
              decay_temperature_step=config.decay_temperature_step,
              decay_temperature_ratio=config.decay_temperature_ratio,
              save_arch_params_frequence=config.save_frequence,
              save_result_path=args.model_save_path,
              decay_temperature_every_epoch=False)

searcher.search(epoch=args.epochs,
                start_w_epoch=config.start_w_epoch,
                log_frequence=args.log_frequence)
