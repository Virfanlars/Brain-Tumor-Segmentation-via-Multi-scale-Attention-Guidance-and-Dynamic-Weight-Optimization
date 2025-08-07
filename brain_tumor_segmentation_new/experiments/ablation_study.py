#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from brain_tumor_segmentation.models.multi_scale_model import MultiScaleModel
from brain_tumor_segmentation.models.losses import DiceLoss, FocalLoss, DynamicWeightedLoss
from brain_tumor_segmentation.data import BrainTumorDataset, create_dataloader, get_transforms
from brain_tumor_segmentation.utils.metrics import calculate_metrics
from brain_tumor_segmentation.utils.logger import setup_logger
from brain_tumor_segmentation.config import config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='脑肿瘤分割消融实验')
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR, help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='results/ablation', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    return parser.parse_args()


def main():
    """主函数，运行消融实验"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger("ablation_study", os.path.join(args.output_dir, "ablation.log"))
    logger.info(f"参数设置: {vars(args)}")
    
    # 加载数据
    transform = get_transforms(mode='test')
    test_dataset = BrainTumorDataset(args.data_dir, mode='test', transform=transform)
    test_loader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 定义消融实验配置
    ablation_configs = [
        {"name": "完整模型", "use_csda": True, "use_dwd": True, "loss": "dynamic"},
        {"name": "无CSDA", "use_csda": False, "use_dwd": True, "loss": "dynamic"},
        {"name": "无DWD", "use_csda": True, "use_dwd": False, "loss": "dynamic"},
        {"name": "固定权重Dice", "use_csda": True, "use_dwd": True, "loss": "dice"},
        {"name": "Focal Loss", "use_csda": True, "use_dwd": True, "loss": "focal"}
    ]
    
    results = {}
    
    # 运行每个消融实验配置
    for config_dict in ablation_configs:
        logger.info(f"运行配置: {config_dict['name']}")
        
        # 创建模型
        model = MultiScaleModel(
            in_channels=1, 
            out_channels=3, 
            use_csda=config_dict['use_csda'],
            use_dwd=config_dict['use_dwd']
        ).to(args.device)
        
        # 加载预训练权重(需要首先训练好这些模型)
        weights_path = os.path.join(
            args.output_dir, 
            f"model_{config_dict['name'].replace(' ', '_')}.pth"
        )
        
        # 注意：这里假设已经提前训练好了各种配置的模型
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=args.device))
            logger.info(f"加载权重: {weights_path}")
        else:
            logger.error(f"权重文件不存在: {weights_path}")
            logger.info("请先训练各种配置的模型，保存为相应的权重文件")
            continue
        
        # 评估模型
        model.eval()
        metrics_all = []
        
        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc=f"评估 {config_dict['name']}"):
                images = images.to(args.device)
                masks = masks.to(args.device)
                
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                
                # 计算指标
                metrics = calculate_metrics(preds.cpu().numpy(), masks.cpu().numpy())
                metrics_all.append(metrics)
        
        # 计算平均指标
        metrics_mean = {key: np.mean([m[key] for m in metrics_all]) for key in metrics_all[0].keys()}
        
        # 记录结果
        results[config_dict['name']] = metrics_mean
        logger.info(f"配置 {config_dict['name']} 的指标: {metrics_mean}")
    
    # 将结果绘制成柱状图
    plot_results(results, os.path.join(args.output_dir, "ablation_results.png"))
    logger.info(f"消融实验结果已保存到 {os.path.join(args.output_dir, 'ablation_results.png')}")


def plot_results(results, save_path):
    """将结果绘制为柱状图"""
    metrics = ['dice', 'iou', 'precision', 'recall', 'specificity']
    configs = list(results.keys())
    
    x = np.arange(len(configs))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    for i, metric in enumerate(metrics):
        values = [results[config][metric] for config in configs]
        ax.bar(x + i * width, values, width, label=metric.capitalize())
    
    ax.set_xlabel('Model Configuration')
    ax.set_ylabel('Score')
    ax.set_title('Ablation Study Results')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(configs)
    ax.legend()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def train_ablation_models(args):
    """训练消融实验的各种模型配置"""
    # 此函数在实际使用时需要实现，用于训练不同配置的模型
    # 这里只是一个示例框架
    pass


if __name__ == '__main__':
    main() 