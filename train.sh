#!/bin/bash

# 设置CUDA可见设备
# export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据实际GPU数量调整

# 设置分布式训练相关环境变量
# export MASTER_PORT=29500
# export MASTER_ADDR=localhost
# export WORLD_SIZE=4  # GPU数量
# export OMP_NUM_THREADS=8

# DEEPSPEED_TIMEOUT=7200 TORCH_NCCL_BLOCKING_WAIT=1 NCCL_DEBUG=INFO TOKENIZERS_PARALLELISM=false NCCL_IB_GID_INDEX=3 NCCL_P2P_DISABLE=1 NCCL_P2P_LEVEL=NVL NCCL_TIMEOUT=7200 \

DEEPSPEED_TIMEOUT=7200 deepspeed --include="localhost:0,1,2,3" \
    parellel_train_step_grpo.py \
    --deepspeed ds_config.json \
    # 2>&1 | tee training.log 