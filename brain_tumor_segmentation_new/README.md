# 基于多尺度注意力引导与动态权重优化的脑肿瘤分割

## 项目简介
本项目提出一种基于多尺度注意力引导与动态权重优化的脑肿瘤分割方法，应用于脑膜瘤、胶质瘤和垂体瘤的精确分割。方法结合了轻量化Mix Vision Transformer、通道-空间双重注意力(CSDA)和动态权重解码器(DWD)等创新设计。

## 数据集
使用包含3064张T1加权对比增强图像的脑肿瘤数据集，来自233名患者：
- 脑膜瘤(708切片)
- 胶质瘤(1426切片)
- 垂体瘤(930切片)

## 主要特性
- **多尺度特征金字塔编码器**：采用轻量化MixViT，通过CSDA模块实现跨尺度特征交互
- **动态权重解码器**：采用渐进式上采样和自适应加权Dice损失，解决类别不平衡问题
- **数据增强策略**：空间变换与强度扰动，增强模型鲁棒性
- **临床分析**：提取形态学特征与生存期进行关联分析

## 目录结构
```
brain_tumor_segmentation/
├── data/              # 数据处理模块
├── models/            # 模型定义
├── utils/             # 工具函数
├── configs/           # 配置文件
├── train/             # 训练脚本
├── evaluate/          # 评估脚本
├── visualize/         # 可视化工具
└── README.md          # 项目说明
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法
1. 数据准备
```bash
python data/prepare_data.py --input_dir /path/to/dataset --output_dir /path/to/processed
```

2. 模型训练
```bash
python train/train.py --config configs/train_config.yaml
```

3. 模型评估
```bash
python evaluate/evaluate.py --model_path /path/to/model --test_data /path/to/test
```

4. 结果可视化
```bash
python visualize/visualize.py --pred_path /path/to/predictions --gt_path /path/to/ground_truth
```