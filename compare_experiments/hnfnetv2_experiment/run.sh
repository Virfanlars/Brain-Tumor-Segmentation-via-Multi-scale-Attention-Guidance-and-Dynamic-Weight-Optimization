#!/bin/bash

# HNF-Netv2模型训练运行脚本


# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=1,2

# 创建输出目录
mkdir -p $(dirname "$0")/output

echo "开始训练HNF-Netv2模型..."
python $(dirname "$0")/train.py --config $(dirname "$0")/config.yaml --gpu $GPU_ID

echo "HNF-Netv2模型训练完成!" 