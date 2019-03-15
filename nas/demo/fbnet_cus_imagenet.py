import argparse
import time
import os
import torch

from ..models.fbnet import FBNetCustom, FBNetCustom_v1_224, FBNet
from ..search.classify_searcher import ClassificationSearcher
from ..utils import _set_file, _logger
from ..datasets.fbnet_data_utils import get_ds

class Config(object):
  alpha = 0.2
  beta = 0.6
  w_lr = 0.1
  w_mom = 0.9
  w_wd = 1e-4
  t_lr = 0.01
  t_wd = 5e-4
  t_beta = (0.9, 0.999)
  model_save_path = '/mnt/data3/zcq/nas/fbnet-pytorch/fbnet/imagenet/'
  start_w_epoch = 2
  train_portion = 0.8
  init_temperature = 5.0
  decay_temperature_ratio = 0.956
  decay_temperature_step = 50
  save_frequence = 50
  num_cls_used = 100

lr_scheduler_params = {
  'logger' : _logger,
  'T_max' : 400,
  'alpha' : 1e-4,
  'warmup_step' : 100,
  't_mul' : 1.5,
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
parser.add_argument('--num-workers', type=int, default=16,
                    help='number of subprocesses used to fetch data, default is 4')
args = parser.parse_args()

args.model_save_path = '%s/%s/' % \
            (config.model_save_path, time.strftime('%Y-%m-%d', time.localtime(time.time())))

if not os.path.exists(args.model_save_path):
  _logger.warn("{} not exists, create it".format(args.model_save_path))
  os.makedirs(args.model_save_path)
_set_file(args.model_save_path + 'log.log')


imagenet_root = '/mnt/data4/zcq/imagenet/train/'
train_queue, val_queue, num_classes = get_ds(args, imagenet_root,
                                num_cls_used=config.num_cls_used)

model = FBNetCustom_v1_224(num_classes,
                           alpha=config.alpha, beta=config.beta)

# TODO(ZhouJ) put this into model or searcher
model.speed_test(torch.randn((1, 3, 224, 224)), verbose=False,
                 device='cuda:' + args.gpus[0])

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
              train_w_ds=train_queue,
              train_arch_ds=val_queue,
              w_sche_cfg=lr_scheduler_params,
              init_temperature=config.init_temperature,
              decay_temperature_ratio=config.decay_temperature_ratio,
              save_arch_params_frequence=config.save_frequence,
              save_result_path=args.model_save_path)

searcher.search(epoch=args.epochs,
                start_w_epoch=config.start_w_epoch,
                log_frequence=args.log_frequence)
