import argparse
import time
import os
import torch

from ..models.proxyless import ProxylessNAS
from ..search.classify_searcher import ClassificationSearcher
from ..datasets.cifar10 import get_cifar10_v1
from ..utils import _set_file, _logger

class Config(object):
  init_theta = 1.0
  alpha = 0.2
  beta = 0.6
  speed_f = './speed_cpu.txt'
  w_lr = 0.1
  w_mom = 0.9
  w_wd = 1e-4
  t_lr = 0.01
  t_wd = 5e-4
  t_beta = (0.9, 0.999)
  model_save_path = '/home1/nas/fbnet-pytorch/'
  total_epoch = 90
  start_w_epoch = 2
  train_portion = 0.8

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
model = ProxylessNAS(10)

# TODO(ZhouJ) put this into model or searcher
model.speed_test(torch.randn((1, 3, 32, 32)))

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
              train_arch_ds=val_ds)

searcher.search(epoch=args.epochs,
                start_w_epoch=5,
                log_frequence=10)
