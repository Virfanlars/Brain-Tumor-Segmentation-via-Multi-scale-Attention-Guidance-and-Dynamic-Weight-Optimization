#!/bin/bash

# 脚本用于运行所有脑肿瘤分割模型的对比实验
# 包括：U-Net、TransUNet和HNF-Netv2

# 设置fold参数
FOLD=${1:-5}  # 默认使用第5折，可以通过命令行参数更改

# 创建结果目录
mkdir -p ../results/comparison

# 设置工作目录为脚本所在目录
cd "$(dirname "$0")"
cd ..

echo "-----------------------------------"
echo "开始训练U-Net模型（第${FOLD}折）..."
echo "-----------------------------------"
python train/train.py --config configs/unet_config.yaml --gpu 0 --fold ${FOLD}
echo "U-Net模型训练完成."

echo "-----------------------------------"
echo "开始训练TransUNet模型（第${FOLD}折）..."
echo "-----------------------------------"
python train/train.py --config configs/transunet_config.yaml --gpu 0 --fold ${FOLD}
echo "TransUNet模型训练完成."

echo "-----------------------------------"
echo "开始训练HNF-Netv2模型（第${FOLD}折）..."
echo "-----------------------------------"
python train/train.py --config configs/hnfnetv2_config.yaml --gpu 0 --fold ${FOLD}
echo "HNF-Netv2模型训练完成."

echo "-----------------------------------"
echo "开始比较所有模型性能（第${FOLD}折）..."
echo "-----------------------------------"
python compare_experiments/compare_results.py \
  --output_dir results/comparison \
  --unet_dir output/unet \
  --transunet_dir output/transunet \
  --hnfnetv2_dir output/hnfnetv2 \
  --fold ${FOLD}

echo "所有实验完成! 结果保存在results/comparison目录" 