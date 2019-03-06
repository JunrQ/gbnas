# Highly custom

import sys
sys.path.insert(0, '/home/zhouchangqing/mxnet/incubator-mxnet_9_17/python')
import mxnet as mx

def get_train_ds(args, kv=None):
    if kv is None:
        rank = 0
        nworker = 1
    else:
        rank, nworker = kv.rank, kv.num_workers
    train = mx.io.ImageRecordIter(
        path_imgrec         = args.train_rec_path,
        label_width         = 1,
        mean_r              = 123.0,
        mean_g              = 116.0,
        mean_b              = 103.0,
        scale               = 0.01,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = tuple(map(int, args.image_shape.split(','))),
        batch_size          = args.batch_size,
        resize_height       = 400,
        resize_width        = 400,
        patch_size          = args.patch_size,
        patch_idx           = args.patch_idx,
        do_aug              = True,
        aug_seq             = 'aug_face',
        FacePatchSize_Main  = 267,
        FacePatchSize_Other = 128,
        PatchFullSize        = 128 if args.image_shape.split(',')[-1] == '108' else 256,
        PatchCropSize        = 108 if args.image_shape.split(',')[-1] == '108' else 224,
        illum_trans_prob    = args.illum_trans_prob,
        gauss_blur_prob     = 0.3,
        motion_blur_prob    = 0.1,
        jpeg_comp_prob    = 0.4,
        res_change_prob    = 0.4,
        hsv_adjust_prob    = args.hsv_adjust_prob,
        preprocess_threads  = args.data_nthreads,
        shuffle             = True,
        # shuffle_chunk_size  = 1024,
        num_parts           = nworker,
        part_index          = rank,
        force2gray = args.force2gray,
        force2color = args.force2color,
        isgray='true' if args.isgray else 'false')
    return train
