#!/usr/bin/env bash

# 54
# CUDA_VISIBLE_DEVICES=1 python test_infer.py 'RAP2' --attr-num 54  --pretrained-model 'rap2_ckpt_max.pth' --test-imgs ./test_images4 --save-path ./test_results4/rap2/
# 51
CUDA_VISIBLE_DEVICES=1 python test_infer.py 'RAP' --attr-num 51  --pretrained-model 'rap_ckpt_max.pth' --test-imgs ./test_images4 --save-path ./test_results4/rap/
# --test-imgs ./test_images
# 35
# CUDA_VISIBLE_DEVICES=1 python test_infer.py 'PETA' --attr-num 35  --pretrained-model 'peta_ckpt_max.pth' --test-imgs ./test_images4 --save-path ./test_results4/peta/
# 26
# CUDA_VISIBLE_DEVICES=1 python test_infer.py 'PA100k' --attr-num 26  --pretrained-model 'pa100k_ckpt_max.pth' --test-imgs ./test_images4 --save-path ./test_results4/pa100k/
# 'PETA', 'PETA_dataset', 'PA100k', 'RAP', 'RAP2'``