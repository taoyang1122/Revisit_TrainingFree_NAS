#!/bin/bash
python evaluate_anglefc.py \
--trainval \
--augtype none \
--repeat 1 \
--score hook_logdet \
--sigma 0.05 \
--nasspace nasbench201 \
--batch_size 256 \
--GPU 0 \
--dataset cifar10 \