

# Gradient based NAS

## Blocks

- weighted_sum : Like fbnet, output of block is the weighted sum of units in the block.
- sample : Like proxyless, output of block is one unit sampled from all the units in the block.
- dag : Like DARTS, treat block as a DAG, every edge's output is the weighted sum of all the operations specified.

## Head

- classification : For classification task, output ce, acc.
- detection : For detection task, highly depend on `mmdetection` and `mmcv`.
- regress(TODO) : For regression task, e.g. landmark
- segmentation(TODO) : For segmentation task

## Models

Assembly class, stack base, block, head.

- darts : snas, but no shared architecture parameters.
- fbnet_faster_rcnn : fbnet + faster rcnn
- proxyless : proxyless, but use reward = - (ce + time cost)

## Search

Do search, train model parameters and architecture parameters iteratively.

## Demos

Assume you are under direcotry `${nas_directory}/`, you can run some demo with following scipts.
NOTE: You may need to modify `path` for *dataset* and *log* in `${nas_directory}/nas/demo/*.py`.

- proxyless

```shell
python -m nas.demo.proxyless_nas_cifar10.py --log-frequence 50 --gpus 0,1 --batch-size 128
```

- fbnet

```shell
python -m nas.demo.fbnet_cus_mxnet_rec.py --gpus 0,1,2,3,4,5,6,7 --log-frequence 100 --batch-size 192
```

- snas

```shell
python -m nas.demo.darts_cifar10.py --gpus 0,1 --log-frequence 50 --batch-size 32
```

- detection

```shell
python -m nas.demo.fbnet_faster_rcnn.py --gpus 0,1,2,3
```
