#!/usr/bin/env bash

# 54
# CUDA_VISIBLE_DEVICES=1 python test_infer.py 'RAP2' --attr-num 54  --pretrained-model 'rap2_ckpt_max.pth' --test-imgs ./test_images4 --save-path ./test_results4/rap2/
# 51
# CUDA_VISIBLE_DEVICES=1 python test_infer.py 'RAP' --attr-num 51  --pretrained-model 'rap_ckpt_max.pth' --test-imgs ./test_images4 --save-path ./test_results4/rap/
# --test-imgs ./test_images
# 35
# CUDA_VISIBLE_DEVICES=1 python test_infer.py 'PETA' --attr-num 35  --pretrained-model 'peta_ckpt_max.pth' --test-imgs ./test_images4 --save-path ./test_results4/peta/
# 26
# CUDA_VISIBLE_DEVICES=1 python test_infer.py 'PA100k' --attr-num 26  --pretrained-model 'pa100k_ckpt_max.pth' --test-imgs ./test_images4 --save-path ./test_results4/pa100k/
# 'PETA', 'PETA_dataset', 'PA100k', 'RAP', 'RAP2'``

# STD-peta
# 2020-12-09_20:15:04_ep_20_bs_256
# CUDA_VISIBLE_DEVICES=1 python test_infer.py 'test' --att-type 'STD' --attr-num 48 --test-imgs ./data/test/images --save-path ./test_result/test/  --pretrained-model '/home/pantengteng/Programs/Strong_Baseline_of_Pedestrian_Attribute_Recognition/exp_result/PETA/2020-12-10_16:17:39_ep_30_bs_64/img_model/ckpt_max.pth' 
# STD-pa100k
# CUDA_VISIBLE_DEVICES=1 python test_infer.py 'pa100k' --att-type 'STD' --attr-num 48 --test-imgs ./data/test/images --save-path ./test_result/test/  --pretrained-model '/home/pantengteng/Programs/Strong_Baseline_of_Pedestrian_Attribute_Recognition/exp_result/PA100k/2020-12-10_20:28:54_ep_30_bs_64/img_model/ckpt_max.pth' 
# 测指标
CUDA_VISIBLE_DEVICES=1 python infer.py 'test' --attr-num 48 --pretrained-model  '/home/pantengteng/Programs/Strong_Baseline_of_Pedestrian_Attribute_Recognition/exp_result/combined/2020-12-12_11:42:11_ep_30_bs_64/img_model/ckpt_max.pth'

# CUDA_VISIBLE_DEVICES=1 python infer.py 'test' --attr-num 48 --pretrained-model '/home/pantengteng/Programs/Strong_Baseline_of_Pedestrian_Attribute_Recognition/exp_result/PA100k/2020-12-10_20:28:54_ep_30_bs_64/img_model/ckpt_max.pth' 
# CUDA_VISIBLE_DEVICES=1 python infer.py 'test' --attr-num 48 --pretrained-model '/home/pantengteng/Programs/Strong_Baseline_of_Pedestrian_Attribute_Recognition/exp_result/PETA/2020-12-10_16:17:39_ep_30_bs_64/img_model/ckpt_max.pth' 