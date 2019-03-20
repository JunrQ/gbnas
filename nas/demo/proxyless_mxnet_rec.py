import argparse
import time
import os
import torch

from ..datasets._utils import get_train_ds
from ..datasets.mxnet_rec import MXIterWrapper

from ..models.proxyless import ProxylessNAS_face
from ..search.classify_searcher import ClassificationSearcher
from ..datasets.mxnet_rec import MXIterWrapper
from ..utils import _set_file, _logger

class Config(object):
  alpha = 100
  w_lr = 0.01
  w_mom = 0.9
  w_wd = 1e-4
  t_lr = 0.01
  t_wd = 5e-4
  t_beta = (0.9, 0.999)
  # model_save_path = '/mnt/data3/zcq/nas/fbnet-pytorch/100w/'
  model_save_path = '/home1/zcq/nas/fbnet-pytorch/proxyless/2k/v1/'
  start_w_epoch = 10
  train_len = 3000 # Number of epoches
  train_portion = 0.8
  valid_len = int(train_len * train_portion)
  init_temperature = 5.0
  decay_temperature_ratio = 0.956
  # decay_temperature_step = 50
  save_frequence = 5000

lr_scheduler_params = {
  'logger' : _logger,
  'T_max' : 400,
  'alpha' : 1e-4,
  'warmup_step' : 100,
  't_mul' : 1.01,
  'lr_mul' : 0.95,
}

config = Config()

parser = argparse.ArgumentParser(description="Train a model with data parallel for base net \
                                and model parallel for classify net.")
parser.add_argument('--batch-size', type=int, default=256,
                    help='training batch size of all devices.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='number of training epochs.')
parser.add_argument('--log-frequence', type=int, default=400,
                    help='log frequence, default is 400')
parser.add_argument('--gpus', type=str, default='0',
                    help='gpus, default is 0')
parser.set_defaults(
    # WebA1
    # num_classes          =  2235656,  #967410,  #
    # num_examples         =  5203228,  #60644986,   #
    # num_classes=967410,  #
    # num_examples=60644986,  #
    # num_classes          =  105381,  #
    # num_examples         =  5544050,   #
    # num_classes = 81968, # 8w reid
    # num_examples = 3551853, # 8w reid

    num_classes = 2000,
    # num_examples = int(107588 / 2),
    image_shape='3,108,108',
    patch_idx=0,
    patch_size=1,
    data_nthreads=16,
    force2gray='false',
    force2color='false',
    illum_trans_prob=0.3,
    hsv_adjust_prob=0.1,
    # train_rec_path       = '/mnt/data3/zhuzhou/image_labels/SrvA1_fn_lb_lmk.rec',
    # train_rec_path='/mnt/data4/zcq/face/recognition/training/imgs/WebA1/train_WebA1_100w.rec',
    # train_rec_path='/mnt/data4/yangling/face/recognition/training/imgs/WebA1/train_WebA1_100w.rec', # 245
    # train_rec_path = '/mnt/data1/yangling_2/face/recognition/training/imgs/WebA1/train_WebA1_100w.rec', # 243
    # train_rec_path = '/home/zhouchangqing/face/recognition/training/imgs/WebA1/train_WebA1_100w.rec',
    # train_rec_path = '/home1/data/face_recognition/face/recognition/training/imgs/WebA1/train_WebA1_100w.rec',
    # train_rec_path       =  '/home1/data/guag/color10W/MsCeleb_SrvA2_fn_lb_lmk_train.rec',
    # train_rec_path = '/home1/data/zhuzhou/MsCeleb_SrvA2_clean/MsCeleb_SrvA2_train_clean_shuffle.rec', # 10w
    # train_rec_path = '/mnt/data4/zcq/10w/zhuzhou/MsCeleb_SrvA2_clean/MsCeleb_SrvA2_train_clean_shuffle.rec', # 10w
    # train_rec_path = '/home1/data/zhuzhou/MsCeleb_SrvA2_clean/SrvA2_train_clean_shuffle.rec', # 2w 
    
    # train_rec_path = '/home1/data/zhuzhou/MsCeleb_SrvA2_clean/MsCeleb_train_clean_reid.rec', # 8w, reid
    # train_rec_path = '/mnt/data4/zcq/10w/zhuzhou/MsCeleb_SrvA2_clean/MsCeleb_train_clean_reid.rec',
    
    # train_rec_path = '/home1/data/zhuzhou/MsCeleb_SrvA2_clean/SrvA2_train_clean_reid.rec', # 2w, reid
    # train_rec_path = '/mnt/data4/zcq/10w/zhuzhou/MsCeleb_SrvA2_clean/SrvA2_train_clean_reid.rec',

    # train_rec_path = '/mnt/data4/zcq/10w/zhuzhou/MsCeleb_SrvA2_clean/MsCeleb_clean1_2w_train_2k.rec', # 2k reid
    train_rec_path = '/home1/data/zhuzhou/MsCeleb_SrvA2_clean/MsCeleb_clean1_2w_train_2k.rec',
    isgray=False,
)
args = parser.parse_args()

args.model_save_path = '%s/%s/' % \
            (config.model_save_path, time.strftime('%Y-%m-%d', time.localtime(time.time())))

if not os.path.exists(args.model_save_path):
  _logger.warn("{} not exists, create it".format(args.model_save_path))
  os.makedirs(args.model_save_path)
_set_file(args.model_save_path + 'log.log')

mx_rec_iter = get_train_ds(args)
wrapper = MXIterWrapper(mx_rec_iter, train_len=config.train_len, valid_len=config.valid_len)
train_ds = wrapper.get_train()
val_ds = wrapper.get_valid()

model = ProxylessNAS_face(args.num_classes, alpha=config.alpha)

# TODO(ZhouJ) put this into model or searcher
model.speed_test(torch.randn((1, 3, 108, 108)), verbose=False,
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
              no_temperature=True,
              save_arch_params_frequence=config.save_frequence,
              save_result_path=args.model_save_path)

searcher.search(epoch=args.epochs,
                start_w_epoch=config.start_w_epoch,
                log_frequence=args.log_frequence)
