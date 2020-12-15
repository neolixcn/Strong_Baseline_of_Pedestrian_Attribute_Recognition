#!/usr/bin/env bash
# ['PETA', 'PETA_dataset', 'PA100k', 'RAP', 'RAP2', 'combined', 'test']
# python train.py PETA --device 0 --train_epoch 30 --batchsize 64 --height 256 --width 192 
python train.py combined --device 0 --train_epoch 30 --batchsize 64 --height 256 --width 192 