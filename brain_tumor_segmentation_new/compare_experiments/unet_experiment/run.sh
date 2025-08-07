#!/bin/bash

# U-Net模型训练运行脚本

# 设置GPU ID
GPU_ID=0

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 创建输出目录
mkdir -p $(dirname "$0")/output

echo "开始训练U-Net模型..."
python $(dirname "$0")/train.py --config $(dirname "$0")/config.yaml --gpu $GPU_ID

echo "U-Net模型训练完成!" 