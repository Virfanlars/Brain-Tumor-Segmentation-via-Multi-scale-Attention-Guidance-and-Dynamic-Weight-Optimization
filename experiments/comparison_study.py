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
import pandas as pd
import seaborn as sns

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from brain_tumor_segmentation.models.multi_scale_model import MultiScaleModel
from brain_tumor_segmentation.models.unet import UNet
from brain_tumor_segmentation.models.transunet import TransUNet
from brain_tumor_segmentation.models.hnfnetv2 import HNFNetv2
from brain_tumor_segmentation.data import BrainTumorDataset, create_dataloader, get_transforms
from brain_tumor_segmentation.utils.metrics import calculate_metrics
from brain_tumor_segmentation.utils.logger import setup_logger
from brain_tumor_segmentation.config import config
from brain_tumor_segmentation.visualize.visualize import difference_visualization


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='脑肿瘤分割方法对比实验')
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR, help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='results/comparison', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--visualize', action='store_true', help='是否生成可视化结果')
    parser.add_argument('--visualize_samples', type=int, default=5, help='生成可视化结果的样本数量')
    return parser.parse_args()


def main():
    """主函数，运行对比实验"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    # 设置日志
    logger = setup_logger("comparison_study", os.path.join(args.output_dir, "comparison.log"))
    logger.info(f"参数设置: {vars(args)}")
    
    # 加载数据
    transform = get_transforms(mode='test')
    test_dataset = BrainTumorDataset(args.data_dir, mode='test', transform=transform)
    test_loader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 定义对比模型
    models = {
        "UNet": {
            "model": UNet(in_channels=1, out_channels=3),
            "weights": "weights/unet.pth"
        },
        "TransUNet": {
            "model": TransUNet(in_channels=1, out_channels=3),
            "weights": "weights/transunet.pth"
        },
        "HNF-Netv2": {
            "model": HNFNetv2(in_channels=1, out_channels=3),
            "weights": "weights/hnfnetv2.pth"
        },
        "提出的方法": {
            "model": MultiScaleModel(in_channels=1, out_channels=3),
            "weights": "weights/multi_scale_model.pth"
        }
    }
    
    results = {}
    visualization_samples = {}
    
    # 随机选择要可视化的样本索引
    if args.visualize:
        vis_indices = np.random.choice(len(test_dataset), args.visualize_samples, replace=False)
        logger.info(f"将可视化测试集中的以下样本: {vis_indices}")
    
    # 评估每个模型
    for model_name, model_info in models.items():
        logger.info(f"评估模型: {model_name}")
        
        model = model_info["model"].to(args.device)
        
        # 加载预训练权重
        weights_path = model_info["weights"]
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=args.device))
            logger.info(f"加载权重: {weights_path}")
        else:
            logger.error(f"权重文件不存在: {weights_path}")
            logger.info(f"请确保 {model_name} 的权重文件存在于 {weights_path}")
            continue
        
        # 评估模型
        model.eval()
        metrics_all = []
        inference_times = []
        
        # 用于可视化的样本
        if args.visualize:
            visualization_samples[model_name] = []
        
        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(test_loader, desc=f"评估 {model_name}")):
                images = images.to(args.device)
                masks = masks.to(args.device)
                
                # 测量推理时间
                start_time = time.time()
                outputs = model(images)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                preds = torch.argmax(outputs, dim=1)
                
                # 计算指标
                metrics = calculate_metrics(preds.cpu().numpy(), masks.cpu().numpy())
                metrics_all.append(metrics)
                
                # 保存可视化样本
                if args.visualize:
                    for batch_idx in range(images.shape[0]):
                        sample_idx = i * args.batch_size + batch_idx
                        if sample_idx in vis_indices:
                            visualization_samples[model_name].append({
                                "image": images[batch_idx].cpu().numpy(),
                                "mask": masks[batch_idx].cpu().numpy(),
                                "pred": preds[batch_idx].cpu().numpy(),
                                "sample_idx": sample_idx
                            })
        
        # 计算平均指标
        metrics_mean = {key: np.mean([m[key] for m in metrics_all]) for key in metrics_all[0].keys()}
        avg_inference_time = np.mean(inference_times) * 1000  # 转换为毫秒
        
        # 记录结果
        metrics_mean["inference_time_ms"] = avg_inference_time
        results[model_name] = metrics_mean
        
        logger.info(f"模型 {model_name} 的指标: {metrics_mean}")
        logger.info(f"模型 {model_name} 的平均推理时间: {avg_inference_time:.2f} ms")
    
    # 如果需要可视化
    if args.visualize and len(visualization_samples) > 0:
        generate_visualizations(visualization_samples, os.path.join(args.output_dir, 'visualizations'))
    
    # 将结果绘制成表格和图表
    plot_results(results, args.output_dir)
    logger.info(f"对比实验结果已保存到 {args.output_dir}")


def plot_results(results, output_dir):
    """将结果绘制为表格和图表"""
    # 创建结果DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')
    
    # 保存结果表格
    df.to_csv(os.path.join(output_dir, 'comparison_results.csv'))
    
    # 绘制条形图
    metrics = ['dice', 'iou', 'precision', 'recall', 'specificity']
    
    plt.figure(figsize=(15, 8))
    df[metrics].plot(kind='bar', figsize=(15, 8))
    plt.title('Performance Metrics Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制推理时间条形图
    plt.figure(figsize=(10, 6))
    plt.bar(df.index, df['inference_time_ms'])
    plt.title('Inference Time Comparison')
    plt.ylabel('Time (ms)')
    plt.xlabel('Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'inference_time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[metrics], annot=True, cmap='viridis', fmt='.3f')
    plt.title('Performance Metrics Heatmap')
    plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_visualizations(visualization_samples, output_dir):
    """为每个模型生成分割可视化结果"""
    model_names = list(visualization_samples.keys())
    
    # 按样本索引对结果进行分组
    samples_by_idx = {}
    
    for model_name in model_names:
        for sample in visualization_samples[model_name]:
            idx = sample["sample_idx"]
            if idx not in samples_by_idx:
                samples_by_idx[idx] = {}
            samples_by_idx[idx][model_name] = sample
    
    # 为每个样本创建可视化
    for idx, models_results in samples_by_idx.items():
        # 确保所有模型都有这个样本的结果
        if len(models_results) != len(model_names):
            continue
        
        # 创建一个大的图，包含所有模型的结果
        fig, axes = plt.subplots(1, len(model_names) + 1, figsize=(5 * (len(model_names) + 1), 5))
        
        # 第一个subplot显示原始图像和真实标签
        sample = list(models_results.values())[0]
        image = sample["image"].squeeze()
        mask = sample["mask"].squeeze()
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image & Ground Truth')
        
        # 叠加真实标签
        mask_rgb = np.zeros((*mask.shape, 3))
        mask_rgb[mask == 1] = [1, 0, 0]  # 红色表示类别1
        mask_rgb[mask == 2] = [0, 1, 0]  # 绿色表示类别2
        
        axes[0].imshow(mask_rgb, alpha=0.3)
        axes[0].axis('off')
        
        # 其余subplot显示每个模型的预测结果
        for i, model_name in enumerate(model_names, 1):
            sample = models_results[model_name]
            pred = sample["pred"].squeeze()
            
            # 创建预测掩码的RGB表示
            pred_rgb = np.zeros((*pred.shape, 3))
            pred_rgb[pred == 1] = [1, 0, 0]  # 红色表示类别1
            pred_rgb[pred == 2] = [0, 1, 0]  # 绿色表示类别2
            
            # 显示原始图像
            axes[i].imshow(image, cmap='gray')
            # 叠加预测
            axes[i].imshow(pred_rgb, alpha=0.3)
            axes[i].set_title(f'{model_name}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{idx}_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 为每个模型创建差异可视化
        for model_name in model_names:
            sample = models_results[model_name]
            image = sample["image"].squeeze()
            mask = sample["mask"].squeeze()
            pred = sample["pred"].squeeze()
            
            # 生成差异可视化
            fig = difference_visualization(image, pred, mask)
            plt.savefig(os.path.join(output_dir, f'sample_{idx}_{model_name}_diff.png'), dpi=300, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    main() 