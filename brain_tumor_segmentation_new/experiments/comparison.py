#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import time
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
import glob

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from models.network import BrainTumorSegNet
from models.unet import UNet
from models.transunet import TransUNet
from models.hnfnetv2 import HNFNetv2
from data.dataset import create_dataloader
from utils.metrics import calculate_metrics
from visualize.visualize import difference_visualization


def parse_args():
    parser = argparse.ArgumentParser(description='Brain Tumor Segmentation Comparison Experiments')
    parser.add_argument('--config', type=str, default='configs/compare_config.yaml', help='Configuration file path')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory, override config setting')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory, override config setting')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization results, override config')
    parser.add_argument('--fold', type=int, default=5, help='要比较的模型折')
    
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def update_config(config, args):
    """根据命令行参数更新配置"""
    if args.data_dir:
        config['data_dir'] = args.data_dir
    
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    if args.visualize:
        config['eval']['visualize'] = True
    
    return config

def create_model(model_name, model_config):
    """Create model based on model name"""
    # 确保输出通道数为1，与训练时一致
    out_channels = 1  # 强制设置为1，确保与训练时一致
    print(f"创建模型: {model_name}, 输出通道数: {out_channels}")
    
    if model_name == "UNet":
        model = UNet(
            in_channels=1,
            out_channels=out_channels,
            features=model_config.get('features', [64, 128, 256, 512])
        )
    elif model_name == "TransUNet":
        model = TransUNet(
            in_channels=1,
            out_channels=out_channels,
            img_size=model_config.get('img_size', 256),
            patch_size=model_config.get('patch_size', 16),
            embed_dim=model_config.get('embed_dim', 768),
            depth=model_config.get('depth', 12),
            n_heads=model_config.get('n_heads', 12),
            features=model_config.get('features', [64, 128, 256, 512])
        )
    elif model_name == "HNF-Netv2":
        model = HNFNetv2(
            in_channels=1,
            out_channels=out_channels,
            features=model_config.get('features', [64, 128, 256, 512])
        )
    elif model_name == "Proposed Method":
        model = BrainTumorSegNet(
            in_channels=1,
            num_classes=out_channels,
            encoder_dims=model_config.get('encoder_dims', [64, 128, 256, 512]),
            decoder_dim=model_config.get('decoder_dim', 128),
            dropout_ratio=model_config.get('dropout_ratio', 0.1)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    return model

def evaluate_model(model, dataloader, device, model_config, visualize=False, sample_indices=None):
    """评估模型性能"""
    model.eval()
    metrics_all = []
    inference_times = []
    
    # 获取模型类型
    model_type = model.__class__.__name__
    print(f"模型类型: {model_type}")
    
    # 判断是否是BrainTumorSegNet（从训练检查点加载的模型）
    is_brain_tumor_segnet = "BrainTumorSegNet" in model_type
    
    # 获取模型特定阈值
    threshold = model_config.get('threshold', 0.5)
    print(f"模型初始阈值设置为: {threshold}")
    
    # 自动阈值搜索选项
    auto_threshold = model_config.get('auto_threshold', False)
    
    # Samples for visualization
    visual_samples = []
    
    # Calculate prediction and ground truth distribution
    pred_pixel_count = 0
    total_pixels = 0
    class_distribution = {}
    
    # 跟踪每个批次的分割性能
    batch_metrics = []
    
    # 添加诊断统计信息
    prediction_stats = {
        'min_values': [],
        'max_values': [],
        'mean_values': [],
        'std_values': [],
        'pred_positive_ratio': [],
        'gt_positive_ratio': [],
        'value_histogram': None
    }
    
    # 创建值直方图统计
    value_bins = np.linspace(0, 1, 21)  # 创建20个值区间
    histogram_counts = np.zeros(20)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Check input data
            if i == 0:
                print(f"输入图像形状: {images.shape}")
                print(f"输入图像范围: [{images.min().item():.4f}, {images.max().item():.4f}]")
                print(f"标签形状: {masks.shape}")
                print(f"标签范围: [{masks.min().item():.4f}, {masks.max().item():.4f}]")
                print(f"标签唯一值: {torch.unique(masks).cpu().numpy()}")
                
                # 收集标签统计信息
                gt_positive = masks.sum().item()
                gt_total = masks.numel()
                gt_positive_ratio = gt_positive / gt_total
                print(f"标签正例像素比例: {gt_positive_ratio:.6f} ({gt_positive}/{gt_total})")
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # 检查输出结果并收集统计信息
            output_min = outputs.min().item()
            output_max = outputs.max().item()
            output_mean = outputs.mean().item()
            output_std = outputs.std().item()
            
            prediction_stats['min_values'].append(output_min)
            prediction_stats['max_values'].append(output_max)
            prediction_stats['mean_values'].append(output_mean)
            prediction_stats['std_values'].append(output_std)
            
            # 对于第一个批次显示更详细的信息
            if i == 0:
                print(f"模型输出形状: {outputs.shape}")
                print(f"输出范围: [{output_min:.6f}, {output_max:.6f}]")
                print(f"输出均值: {output_mean:.6f}")
                print(f"输出标准差: {output_std:.6f}")
                
                # 为第一个批次创建值分布直方图
                if outputs.size(1) == 1:
                    # 对于2D值分布，创建直方图
                    flat_outputs = outputs.detach().cpu().numpy().flatten()
                    hist, _ = np.histogram(flat_outputs, bins=value_bins)
                    histogram_counts += hist
                    
                    # 输出直方图值分布
                    print("输出值分布直方图:")
                    for j in range(len(hist)):
                        bin_start = value_bins[j]
                        bin_end = value_bins[j+1]
                        print(f"  [{bin_start:.2f}-{bin_end:.2f}]: {hist[j]}")
                
                if outputs.size(1) > 1:
                    # 如果是多类别，查看每个类别的输出
                    print(f"多类别输出的前5个值: {outputs[0, :, 0, 0]}")
                else:
                    # 显示单个输出的采样值
                    print(f"第一个样本的部分输出值: {outputs[0, 0, :5, :5]}")
                
                # 检查输出是否已经经过激活处理
                if outputs.size(1) == 1:
                    # 如果输出范围已经在[0,1]之间，可能已经应用了sigmoid
                    is_already_sigmoid = output_min >= 0 and output_max <= 1
                    if is_already_sigmoid:
                        print("检测到输出值范围在[0,1]之间，似乎已经应用了sigmoid激活")
                
                # 始终使用sigmoid处理输出 - 无论是否已应用，以保持一致性
                if outputs.size(1) == 1:
                    probs = torch.sigmoid(outputs)
                    print(f"强制应用sigmoid后，处理后范围: [{probs.min().item():.6f}, {probs.max().item():.6f}]")
                    
                    # 检查sigmoid是否有效果
                    if abs(probs.min().item() - probs.max().item()) < 0.01:
                        print("警告: sigmoid后的输出范围非常窄，这可能表明模型权重加载有问题或未正确训练")
                    
                    # 在阈值处检查值的分布
                    below_thresh = (probs <= threshold).float().sum().item()
                    above_thresh = (probs > threshold).float().sum().item()
                    total = below_thresh + above_thresh
                    print(f"阈值({threshold})分布: 低于={below_thresh/total:.6f}, 高于={above_thresh/total:.6f}")
                        
                # 自动搜索最佳阈值（仅针对第一个批次）
                if auto_threshold and outputs.size(1) == 1 and i == 0:
                    # 确保我们有概率值
                    print("\n开始自动搜索最佳阈值...")
                    # 在一定范围内测试不同阈值
                    best_dice = -1
                    best_thresh = threshold
                    
                    # 根据输出范围确定搜索范围
                    prob_min = probs.min().item()
                    prob_max = probs.max().item()
                    print(f"概率值范围: [{prob_min:.6f}, {prob_max:.6f}]")
                    
                    # 检查概率值范围是否太窄
                    if prob_max - prob_min < 0.05:
                        print("警告: 概率值范围非常窄，可能是模型问题，设置更宽的搜索范围")
                        search_min = max(0.01, prob_min - 0.1)
                        search_max = min(0.99, prob_max + 0.1)
                    else:
                        # 设置搜索范围，确保涵盖概率范围
                        search_min = max(0.1, prob_min - 0.05)
                        search_max = min(0.9, prob_max + 0.05)
                    
                    print(f"阈值搜索范围: [{search_min:.4f}, {search_max:.4f}]")
                    
                    # 从低到高尝试不同阈值
                    threshold_candidates = np.linspace(search_min, search_max, 20)
                    for test_thresh in threshold_candidates:
                        test_preds = (probs > test_thresh).float()
                        
                        # 计算该阈值下的Dice系数
                        batch_pred = test_preds.cpu().numpy()
                        batch_mask = masks.cpu().numpy()
                        if batch_mask.shape[1] == 1:
                            batch_mask = batch_mask.squeeze(1)
                            batch_pred = batch_pred.squeeze(1)
                        
                        # 调用accuracy_score计算指标
                        dice_score = calculate_metrics(batch_pred, batch_mask)['dice']
                        
                        print(f"阈值 {test_thresh:.4f} -> Dice = {dice_score:.4f}")
                        
                        if dice_score > best_dice:
                            best_dice = dice_score
                            best_thresh = test_thresh
                    
                    # 更新为最佳阈值
                    threshold = best_thresh
                    print(f"\n选择最佳阈值: {threshold:.4f} (Dice = {best_dice:.4f})")
            
            # 计算预测结果
            if outputs.size(1) == 1:
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).float()
            else:
                # 多类别情况
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1, keepdim=True).float()
            
            # 收集预测正例比例
            pred_positive = preds.sum().item()
            pred_total = preds.numel()
            pred_positive_ratio = pred_positive / pred_total
            prediction_stats['pred_positive_ratio'].append(pred_positive_ratio)
            
            # 收集标签正例比例
            gt_positive = masks.sum().item()
            gt_total = masks.numel()
            gt_positive_ratio = gt_positive / gt_total
            prediction_stats['gt_positive_ratio'].append(gt_positive_ratio)
            
            # 计算此批次的指标
            batch_preds = preds.cpu().numpy()
            batch_masks = masks.cpu().numpy()
            
            if batch_masks.shape[1] == 1:
                batch_masks = batch_masks.squeeze(1)
                batch_preds = batch_preds.squeeze(1)
            
            # 计算此批次的性能指标
            batch_metrics.append(calculate_metrics(batch_preds, batch_masks))
            
            # 收集可视化样本
            if visualize and sample_indices is not None:
                for idx in sample_indices:
                    if idx < len(dataloader.dataset) and i*dataloader.batch_size <= idx < (i+1)*dataloader.batch_size:
                        local_idx = idx - i*dataloader.batch_size
                        if local_idx < images.size(0):
                            visual_samples.append({
                                'image': images[local_idx].cpu().numpy(),
                                'mask': masks[local_idx].cpu().numpy(),
                                'pred': preds[local_idx].cpu().numpy(),
                                'prob': probs[local_idx].cpu().numpy()
                        })
    
    # 计算平均指标
    metrics = {}
    for key in batch_metrics[0].keys():
        metrics[key] = np.mean([b[key] for b in batch_metrics])
    
    # 添加阈值和推理时间到指标
    metrics['threshold'] = threshold
    metrics['inference_time'] = np.mean(inference_times)
    
    # 计算并输出诊断统计信息
    print("\n模型输出诊断统计:")
    print(f"平均最小值: {np.mean(prediction_stats['min_values']):.6f}")
    print(f"平均最大值: {np.mean(prediction_stats['max_values']):.6f}")
    print(f"平均输出均值: {np.mean(prediction_stats['mean_values']):.6f}")
    print(f"平均标准差: {np.mean(prediction_stats['std_values']):.6f}")
    print(f"预测正例像素平均比例: {np.mean(prediction_stats['pred_positive_ratio']):.6f}")
    print(f"真实标签正例像素平均比例: {np.mean(prediction_stats['gt_positive_ratio']):.6f}")
    
    # 检查预测与真实标签的差异
    pred_gt_ratio = np.mean(prediction_stats['pred_positive_ratio']) / np.mean(prediction_stats['gt_positive_ratio']) if np.mean(prediction_stats['gt_positive_ratio']) > 0 else float('inf')
    print(f"预测/真实正例比例: {pred_gt_ratio:.2f}")
    
    if pred_gt_ratio > 10 or pred_gt_ratio < 0.1:
        print(f"警告: 预测与真实标签的正例比例相差很大。这可能表明模型预测异常。")
        if pred_gt_ratio > 10:
            print("模型预测了太多的正例（过度预测）。这可能是因为模型未正确训练或权重不匹配。")
        else:
            print("模型预测了太少的正例（预测不足）。这可能是因为模型过于保守或阈值过高。")
    
    # 输出值分布对比
    if len(value_bins) > 1:
        print("\n激活前输出值分布:")
        for j in range(len(histogram_counts)):
            bin_start = value_bins[j]
            bin_end = value_bins[j+1]
            print(f"  [{bin_start:.2f}-{bin_end:.2f}]: {histogram_counts[j]}")

    return metrics, visual_samples

def create_visualizations(samples_by_model, output_dir):
    """创建可视化对比结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有样本索引
    all_samples = {}
    for model_name, samples in samples_by_model.items():
        for sample in samples:
            sample_idx = sample['idx']
            if sample_idx not in all_samples:
                all_samples[sample_idx] = {}
            all_samples[sample_idx][model_name] = sample
    
    # 为每个样本创建可视化
    for sample_idx, model_results in all_samples.items():
        print(f"创建样本 {sample_idx} 的可视化结果...")
        
        # 获取第一个模型的样本来提取图像和真实标签
        first_model = list(model_results.keys())[0]
        image = model_results[first_model]['image']
        mask = model_results[first_model]['mask']
        
        # 创建基础图:原始图像和真实标签
        n_models = len(model_results)
        fig, axes = plt.subplots(2, n_models + 1, figsize=(4 * (n_models + 1), 8))
        
        # 显示原始图像
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')
        
        # 显示真实标签
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('真实标签')
        axes[1, 0].axis('off')
        
        # 显示每个模型的预测结果
        for i, (model_name, sample) in enumerate(model_results.items()):
            pred = sample['pred']
            
            # 显示预测结果
            col = i + 1
            
            # 检查是否有概率图可用
            if 'prob' in sample and sample['prob'] is not None:
                # 显示概率图
                axes[0, col].imshow(sample['prob'], cmap='jet', vmin=0, vmax=1)
                axes[0, col].set_title(f'{model_name}\n概率图')
            else:
                # 显示二值预测结果
                axes[0, col].imshow(pred, cmap='gray')
                axes[0, col].set_title(f'{model_name}\n预测')
            axes[0, col].axis('off')
            
            # 显示差异图:绿色=真正例,红色=假正例,蓝色=假负例
            diff_map = np.zeros((*pred.shape, 3))
            # 真正例(绿色)
            diff_map[..., 1] = np.logical_and(pred > 0, mask > 0)
            # 假正例(红色)
            diff_map[..., 0] = np.logical_and(pred > 0, mask == 0)
            # 假负例(蓝色)
            diff_map[..., 2] = np.logical_and(pred == 0, mask > 0)
            
            axes[1, col].imshow(diff_map)
            axes[1, col].set_title(f'{model_name}\n差异图')
            axes[1, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{sample_idx}.png'), dpi=200, bbox_inches='tight')
        plt.close()
        
        # 如果有概率图，为每个模型创建阈值分析图
        for model_name, sample in model_results.items():
            # 检查是否有概率图可用
            if 'prob' in sample and sample['prob'] is not None:
                pred_prob = sample['prob'].squeeze()
                
                # 多行显示：原图+标签、概率图、不同阈值的预测结果
                thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
                fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                
                # 第一行: 原图、真实标签、概率图
                axes[0, 0].imshow(image, cmap='gray')
                axes[0, 0].set_title('原始图像')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(mask, cmap='gray')
                axes[0, 1].set_title('真实标签')
                axes[0, 1].axis('off')
                
                im = axes[0, 2].imshow(pred_prob, cmap='jet', vmin=0, vmax=1)
                axes[0, 2].set_title('概率图')
                axes[0, 2].axis('off')
                plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
                
                # 第二行: 不同阈值的预测结果
                for i, thresh in enumerate(thresholds):
                    if i >= 3:  # 只显示前3个阈值
                        break
                    
                    pred_thresh = (pred_prob > thresh).astype(np.uint8)
                    axes[1, i].imshow(pred_thresh, cmap='gray')
                    axes[1, i].set_title(f'阈值 = {thresh}')
                    axes[1, i].axis('off')
                    
                    # 计算该阈值下的Dice
                    intersection = np.sum(pred_thresh * mask)
                    dice = (2.0 * intersection) / (np.sum(pred_thresh) + np.sum(mask) + 1e-8)
                    axes[1, i].set_xlabel(f'Dice = {dice:.4f}')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'sample_{sample_idx}_{model_name}_thresholds.png'), 
                            dpi=200, bbox_inches='tight')
            plt.close()
    
    print(f"可视化结果已保存至: {output_dir}")

def plot_comparison_results(results, output_dir):
    """Plot comparison experiment results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'comparison_results.csv')
    df.to_csv(csv_path)
    print(f"Results saved to: {csv_path}")
    
    # 检查是否有阈值信息
    has_threshold_info = 'threshold' in df.columns
    threshold_note = ""
    if has_threshold_info:
        threshold_note = "\n(注意: 不同模型使用了不同阈值)"
        
    # Plot performance metrics bar chart
    metrics = ['dice', 'iou', 'precision', 'recall', 'specificity']
    plt.figure(figsize=(12, 8))
    df[metrics].plot(kind='bar', figsize=(12, 8))
    plt.title(f'Performance Metrics Comparison{threshold_note}')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save chart
    metrics_plot_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot inference time bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(df.index, df['inference_time'], color='salmon')
    plt.title('Inference Time Comparison')
    plt.ylabel('Time (ms)')
    plt.xlabel('Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(df['inference_time']):
        plt.text(i, v + 0.5, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    
    # Save chart
    time_plot_path = os.path.join(output_dir, 'inference_time_comparison.png')
    plt.savefig(time_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[metrics], annot=True, cmap='viridis', fmt='.3f')
    plt.title(f'Performance Metrics Heatmap{threshold_note}')
    plt.tight_layout()
    
    # Save heatmap
    heatmap_path = os.path.join(output_dir, 'metrics_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 如果有阈值信息，创建阈值对比图
    if has_threshold_info:
        plt.figure(figsize=(10, 6))
        plt.bar(df.index, df['threshold'], color='lightgreen')
        plt.title('Threshold Values Used for Each Model')
        plt.ylabel('Threshold')
        plt.xlabel('Model')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(df['threshold']):
            plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
        
        plt.tight_layout()
        
        # Save chart
        threshold_plot_path = os.path.join(output_dir, 'threshold_comparison.png')
        plt.savefig(threshold_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Charts saved to: {output_dir}")

def load_model_from_checkpoint(checkpoint_path, device):
    """直接从训练的checkpoint加载模型，确保结构与训练时完全一致"""
    print(f"从训练检查点直接加载模型: {checkpoint_path}")
    
    # 首先加载checkpoint检查内部结构
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 检查权重的内部结构以确定真实的模型类型
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # 分析state_dict键名确定真实模型类型
    keys = list(state_dict.keys())
    # 移除可能的"model."前缀以方便分析
    clean_keys = [k[6:] if k.startswith('model.') else k for k in keys]
    
    # 根据权重结构判断模型类型
    if any('encoder.stages' in k for k in clean_keys):
        actual_model_type = "Proposed Method"  # BrainTumorSegNet
    elif any('encoder.0.conv.0.weight' in k for k in clean_keys):
        actual_model_type = "UNet"
    elif any('transformer.resblocks' in k for k in clean_keys):
        actual_model_type = "TransUNet"
    elif any('hnfnet' in k.lower() for k in clean_keys):
        actual_model_type = "HNF-Netv2"
    else:
        # 回退到基于文件名的检测
        if "transunet" in checkpoint_path.lower():
            actual_model_type = "TransUNet"
        elif "hnfnetv2" in checkpoint_path.lower():
            actual_model_type = "HNF-Netv2"
        elif "unet" in checkpoint_path.lower():
            actual_model_type = "UNet"
        else:
            actual_model_type = "Proposed Method"
    
    print(f"通过权重结构分析，确定模型类型为: {actual_model_type}")
    
    # 创建对应类型的模型
    if actual_model_type == "TransUNet":
        model = TransUNet(in_channels=1, out_channels=1)
    elif actual_model_type == "HNF-Netv2":
        model = HNFNetv2(in_channels=1, out_channels=1)
    elif actual_model_type == "UNet":
        model = UNet(in_channels=1, out_channels=1)
    else:
        # Proposed Method - BrainTumorSegNet
        sys.path.append(os.path.join(root_dir, 'train'))
        from trainer import BrainTumorSegTrainer
        config = {
            'in_channels': 1,
            'num_classes': 1,
            'encoder_dims': [64, 128, 256, 512],
            'decoder_dim': 128,
            'dropout_ratio': 0.1
        }
        # 创建训练器实例
        trainer_model = BrainTumorSegTrainer(config)
        # 实际我们只需要基础模型
        model = trainer_model.model
    
    # 处理state_dict以匹配模型结构
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # 移除"model."前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if actual_model_type == "Proposed Method" and k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        
        # 尝试严格加载
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print(f"严格模式加载权重成功: {checkpoint_path}")
        except Exception as strict_e:
            # 记录错误信息以便分析
            print(f"严格模式加载失败，错误: {str(strict_e)}")
            print(f"缺失的键: {strict_e.missing_keys[:5]}..." if hasattr(strict_e, 'missing_keys') and len(strict_e.missing_keys) > 0 else "无缺失键")
            print(f"意外的键: {strict_e.unexpected_keys[:5]}..." if hasattr(strict_e, 'unexpected_keys') and len(strict_e.unexpected_keys) > 0 else "无意外键")
            
            # 如果存在大量缺失或意外键，说明模型结构与权重不匹配
            serious_mismatch = False
            if hasattr(strict_e, 'missing_keys') and hasattr(strict_e, 'unexpected_keys'):
                total_keys = len(new_state_dict)
                missing_ratio = len(strict_e.missing_keys) / total_keys if total_keys > 0 else 0
                unexpected_ratio = len(strict_e.unexpected_keys) / total_keys if total_keys > 0 else 0
                
                if missing_ratio > 0.5 or unexpected_ratio > 0.5:
                    serious_mismatch = True
                    
            if serious_mismatch:
                raise ValueError(f"模型({actual_model_type})与权重严重不匹配，请检查权重文件是否正确: {checkpoint_path}")
            
            print(f"尝试非严格模式加载...")
            model.load_state_dict(new_state_dict, strict=False)
            print(f"非严格模式加载权重成功: {checkpoint_path}")
    else:
        # 常规模型权重文件
        try:
            model.load_state_dict(checkpoint, strict=True)
            print(f"严格模式加载权重成功: {checkpoint_path}")
        except Exception as e:
            print(f"严格模式加载失败，尝试非严格模式: {str(e)}")
            model.load_state_dict(checkpoint, strict=False)
            print(f"非严格模式加载权重成功: {checkpoint_path}")
    
    # 将模型设为评估模式并返回
    model = model.to(device)
    model.eval()
    
    # 直接返回模型，不添加额外激活层
    return model

def validate_config(config):
    """验证配置文件，确保模型类型与权重文件匹配"""
    models_config = config.get('models', {})
    valid = True
    
    print("\n验证配置文件...")
    for model_key, model_cfg in models_config.items():
        model_name = model_cfg.get('name')
        weight_path = model_cfg.get('weight_path')
        
        if not model_name:
            print(f"错误: 模型 {model_key} 缺少名称")
            valid = False
            continue
            
        if not weight_path:
            print(f"错误: 模型 {model_key} 缺少权重文件路径")
            valid = False
            continue
            
        if not os.path.exists(weight_path):
            print(f"警告: 模型 {model_key} 的权重文件不存在: {weight_path}")
            valid = False
            continue
            
        # 验证权重文件格式
        try:
            weights = torch.load(weight_path, map_location='cpu')
            if not isinstance(weights, dict):
                print(f"警告: 模型 {model_key} 的权重文件格式不是字典: {weight_path}")
                
            # 检查是否有状态字典
            state_dict = weights.get('state_dict', weights)
            if not state_dict or not isinstance(state_dict, dict):
                print(f"警告: 模型 {model_key} 的权重文件没有有效的状态字典: {weight_path}")
                
            # 检查键名以判断真实模型类型
            keys = list(state_dict.keys())
            clean_keys = [k[6:] if k.startswith('model.') else k for k in keys]
            
            # 检测实际模型类型
            if any('encoder.stages' in k for k in clean_keys):
                actual_type = "Proposed Method"  # BrainTumorSegNet
            elif any('encoder.0.conv.0.weight' in k for k in clean_keys):
                actual_type = "UNet"
            elif any('transformer.resblocks' in k for k in clean_keys):
                actual_type = "TransUNet"
            elif any('hnfnet' in k.lower() for k in clean_keys):
                actual_type = "HNF-Netv2"
            else:
                # 尝试从文件名判断
                if "transunet" in weight_path.lower():
                    actual_type = "TransUNet"
                elif "hnfnetv2" in weight_path.lower():
                    actual_type = "HNF-Netv2"
                elif "unet" in weight_path.lower():
                    actual_type = "UNet"
                else:
                    actual_type = "Unknown"
            
            # 检查是否匹配
            if model_name != actual_type and not (model_name == "BrainTumorSegNet" and actual_type == "Proposed Method"):
                print(f"警告: 模型 {model_key} 类型 ({model_name}) 与权重文件实际类型 ({actual_type}) 不匹配")
                print(f"  - 文件: {weight_path}")
                
                # 打印部分键名作为参考
                print(f"  - 权重文件键名示例: {keys[:3]}...")
                valid = False
            else:
                print(f"验证通过: 模型 {model_key} ({model_name}) 与权重文件匹配")
                
        except Exception as e:
            print(f"错误: 无法验证模型 {model_key} 的权重文件: {str(e)}")
            valid = False
    
    return valid

def find_best_checkpoint(checkpoint_dir):
    """查找指定目录中具有最高验证Dice的检查点"""
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"找不到检查点目录: {checkpoint_dir}")
        
    # 寻找最佳检查点（基于文件名中的验证Dice值）
    best_dice = 0.0
    best_ckpt = None
    
    for ckpt_file in glob.glob(os.path.join(checkpoint_dir, "*.ckpt")):
        if 'last' in ckpt_file:
            continue  # 跳过last检查点
            
        try:
            # 从文件名中提取Dice值，格式通常是epoch=XX-val/dice=0.XXXX.ckpt
            dice_str = os.path.basename(ckpt_file).split('-')[1].split('=')[1].split('.ckpt')[0]
            dice = float(dice_str)
            
            if dice > best_dice:
                best_dice = dice
                best_ckpt = ckpt_file
        except:
            continue
            
    # 如果没找到有效的检查点，使用last.ckpt
    if best_ckpt is None:
        best_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
        if not os.path.exists(best_ckpt):
            raise FileNotFoundError(f"在{checkpoint_dir}中找不到有效的检查点")
    
    print(f"选择检查点: {best_ckpt}, Dice: {best_dice:.4f}")
    return best_ckpt

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config = update_config(config, args)
    
    # 验证配置
    config_valid = validate_config(config)
    if not config_valid:
        print("\n警告: 配置验证发现问题。是否继续? (y/n)")
        user_choice = input().lower().strip()
        if user_choice != 'y':
            print("退出程序")
            return
        print("用户确认继续执行...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    if config['eval']['visualize']:
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    # Create test data loader
    test_list = os.path.join(config['data_dir'], f"fold_{config['fold']}", 'test.txt')
    test_loader = create_dataloader(
        test_list,
        batch_size=config['batch_size'],
        mode='test',
        crop_size=config['img_size'],
        center_crop=config['center_crop'],
        num_workers=config['num_workers']
    )
    
    # Generate sample indices for visualization
    if config['eval']['visualize']:
        num_samples = len(test_loader.dataset)
        vis_samples = np.random.choice(
            num_samples, 
            min(config['eval']['visualize_samples'], num_samples), 
            replace=False
        )
        print(f"Will visualize these samples from test set: {vis_samples}")
    else:
        vis_samples = None
    
    # 评估每个模型
    results = {}
    samples_by_model = {}
    
    for model_key, model_config in config['models'].items():
        model_name = model_config['name']
        print(f"\n评估模型: {model_name}")
        
        # 获取权重路径
        weight_path = model_config['weight_path']
        
        # 检查权重文件是否匹配模型类型
        print(f"检查权重文件与模型类型的匹配性: {weight_path}")
        if not os.path.exists(weight_path):
            print(f"警告: 找不到权重文件 {weight_path}，跳过此模型评估")
            continue
            
        try:
            # 验证权重文件的结构
            weights = torch.load(weight_path, map_location=device)
            state_dict = weights.get('state_dict', weights)
            
            # 检查权重键名以判断真实模型类型
            keys = list(state_dict.keys())
            clean_keys = [k[6:] if k.startswith('model.') else k for k in keys]
            
            # 基于权重结构判断模型类型
            if any('encoder.stages' in k for k in clean_keys):
                actual_type = "Proposed Method"  # BrainTumorSegNet
            elif any('encoder.0.conv.0.weight' in k for k in clean_keys):
                actual_type = "UNet"
            elif any('transformer.resblocks' in k for k in clean_keys):
                actual_type = "TransUNet"
            elif any('hnfnet' in k.lower() for k in clean_keys):
                actual_type = "HNF-Netv2"
            else:
                # 如果无法确定，使用文件名
                if "transunet" in weight_path.lower():
                    actual_type = "TransUNet"
                elif "hnfnetv2" in weight_path.lower():
                    actual_type = "HNF-Netv2"
                elif "unet" in weight_path.lower():
                    actual_type = "UNet"
                else:
                    actual_type = "Proposed Method"
            
            # 检查权重类型与指定的模型类型是否匹配
            if actual_type != model_name and not (actual_type == "Proposed Method" and model_name == "BrainTumorSegNet"):
                print(f"警告: 权重文件类型({actual_type})与指定的模型({model_name})不匹配")
                print(f"是否要继续使用这个权重文件? (y/n)")
                user_choice = input().lower().strip()
                if user_choice != 'y':
                    print(f"跳过此模型评估")
                    continue
                else:
                    print(f"用户确认继续使用该权重文件")
            
            # 基于实际模型类型加载模型
            try:
                # 使用统一的方式从检查点加载模型，保持架构一致性
                model = load_model_from_checkpoint(weight_path, device)
                print(f"从训练检查点直接加载{model_name}成功")
                
                # 应用模型专用的最优阈值
                model_config['threshold'] = 0.5  # 统一使用训练时的阈值
                print(f"使用训练时的阈值: {model_config['threshold']}")
                
                # 确保自动阈值搜索关闭
                model_config['auto_threshold'] = False
                
            except Exception as e:
                print(f"从检查点加载模型失败，错误: {str(e)}")
                print("尝试使用常规方式创建模型...")
                
                # 备用方案：使用原有方式创建和加载模型
                model = create_model(model_name, model_config)
                model = model.to(device)
                
                # 添加缺失的激活函数
                if model_name in ["UNet", "TransUNet", "HNF-Netv2"]:
                    # 为模型添加sigmoid激活层
                    class ModelWithActivation(torch.nn.Module):
                        def __init__(self, base_model):
                            super().__init__()
                            self.base_model = base_model
                            
                        def forward(self, x):
                            x = self.base_model(x)
                            return torch.sigmoid(x)
                    
                    model = ModelWithActivation(model)
                    print(f"为{model_name}添加了sigmoid激活层")
                
                # 加载预训练权重，使用严格模式尝试加载
                try:
                    if 'state_dict' in weights:
                        # 从Lightning检查点提取模型状态字典
                        state_dict = weights['state_dict']
                        # 移除"model."前缀
                        new_state_dict = {}
                        for k, v in state_dict.items():
                            if k.startswith('model.'):
                                new_state_dict[k[6:]] = v
                            else:
                                new_state_dict[k] = v
                        
                        try:
                            model.load_state_dict(new_state_dict, strict=True)
                            print(f"严格模式加载权重成功: {weight_path}")
                        except Exception as strict_e:
                            print(f"严格模式加载失败: {str(strict_e)}")
                            
                            # 检查不匹配的严重程度
                            if hasattr(strict_e, 'missing_keys') and hasattr(strict_e, 'unexpected_keys'):
                                if len(strict_e.missing_keys) > 20 or len(strict_e.unexpected_keys) > 20:
                                    print("警告: 大量键不匹配，可能导致性能问题")
                                    
                                # 显示部分不匹配的键以便调试
                                if strict_e.missing_keys:
                                    print(f"缺失的键(前5个): {strict_e.missing_keys[:5]}")
                                if strict_e.unexpected_keys:
                                    print(f"意外的键(前5个): {strict_e.unexpected_keys[:5]}")
                            
                            # 使用非严格模式
                            model.load_state_dict(new_state_dict, strict=False)
                            print(f"非严格模式加载权重成功: {weight_path}")
                    else:
                        # 常规模型权重文件
                        try:
                            model.load_state_dict(weights, strict=True)
                            print(f"严格模式加载权重成功: {weight_path}")
                        except Exception:
                            model.load_state_dict(weights, strict=False)
                            print(f"非严格模式加载权重成功: {weight_path}")
                except Exception as e:
                    print(f"加载权重失败: {str(e)}")
                    print("警告: 将使用未初始化的模型，结果可能不可靠")
                
                # 为这些模型设置较低的阈值
                model_config['threshold'] = 0.07  # 根据输出范围设置较低的阈值
                print(f"设置较低的阈值: {model_config['threshold']}")
        
             # 评估模型
            metrics, visual_samples = evaluate_model(
                model,
                test_loader,
                device,
                model_config,
            visualize=config['eval']['visualize'],
            sample_indices=vis_samples
             )
        
             # 存储结果
            results[model_name] = metrics
            if config['eval']['visualize']:
                samples_by_model[model_name] = visual_samples
        
            # 打印详细指标
            print(f"\n{model_name}评估结果:")
            for metric, value in metrics.items():
                if metric in ['dice', 'iou', 'precision', 'recall', 'specificity', 'hd95']:
                    print(f"{metric}: {value:.4f}")
            
            print(f"使用阈值: {metrics.get('threshold', 'N/A')}")
            print(f"推理时间: {metrics.get('inference_time', 0)*1000:.2f} ms")
            
        except Exception as e:
            print(f"处理模型 {model_name} 时发生错误: {str(e)}")
            print(f"跳过此模型并继续其他模型的评估")
    
    # 汇总所有模型的dice得分，便于比较
    print("\n模型性能比较汇总:")
    for model_name, metrics in results.items():
        dice = metrics.get('dice', 0)
        iou = metrics.get('iou', 0)
        inference_time = metrics.get('inference_time', 0) * 1000  # 转换为毫秒
        print(f"{model_name}: Dice={dice:.4f}, IoU={iou:.4f}, 推理时间={inference_time:.2f}ms")
    
    # 创建性能对比表格
    performance_df = []
    for model_name, metrics in results.items():
        model_row = {'模型': model_name}
        for metric, value in metrics.items():
            if metric in ['dice', 'iou', 'precision', 'recall', 'specificity', 'hd95']:
                model_row[metric] = round(value, 4)
        model_row['inference_time(ms)'] = round(metrics.get('inference_time', 0) * 1000, 2)
        performance_df.append(model_row)
    
    # 转换为DataFrame并保存
    if performance_df:
        performance_table = pd.DataFrame(performance_df)
        performance_table_path = os.path.join(output_dir, 'comparison_results.csv')
        performance_table.to_csv(performance_table_path, index=False)
        print(f"\n性能比较表已保存到: {performance_table_path}")
        print("\n性能比较:")
        print(performance_table)
    
    # Visualize results
    if config['eval']['visualize'] and samples_by_model:
        print("\n生成可视化结果...")
        create_visualizations(samples_by_model, vis_dir)
    
    # Plot and save results
    print("\n生成对比图表...")
    plot_comparison_results(results, output_dir)
    
    print("\n对比实验完成!")

if __name__ == "__main__":
    main() 