#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

# 导入原始项目的模块
from train.trainer import BrainTumorSegTrainer
from data.dataset import BrainTumorDataset, get_transforms, get_cross_validation_folds

def parse_args():
    parser = argparse.ArgumentParser(description='训练脑肿瘤分割模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--fold', type=int, default=None, help='指定训练的折，None表示训练所有折')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--gpu', type=str, default='0', help='使用的GPU ID')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def prepare_data(config, fold):
    """准备数据集"""
    # 获取数据集路径
    data_dir = config['data_dir']
    
    # 获取图像尺寸
    img_size = config.get('img_size', 256)
    
    # 获取数据集划分
    train_indices, val_indices, test_indices = get_cross_validation_folds(
        data_dir=data_dir,
        fold=fold,
        n_folds=config.get('fold', 5),
        seed=config.get('seed', 42)
    )
    
    # 获取数据增强转换
    train_transform, val_transform = get_transforms(
        img_size=img_size,
        center_crop=config.get('center_crop', True)
    )
    
    # 创建数据集
    train_dataset = BrainTumorDataset(
        data_dir=data_dir,
        indices=train_indices,
        transform=train_transform,
        phase='train'
    )
    
    val_dataset = BrainTumorDataset(
        data_dir=data_dir,
        indices=val_indices,
        transform=val_transform,
        phase='val'
    )
    
    test_dataset = BrainTumorDataset(
        data_dir=data_dir,
        indices=test_indices,
        transform=val_transform,
        phase='test'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        num_workers=config.get('num_workers', 8),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=config.get('num_workers', 8),
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=config.get('num_workers', 8),
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def train_model_fold(config, fold):
    """训练单个fold"""
    # 设置fold输出目录
    output_dir = os.path.join(config['output_dir'], f'fold_{fold}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 更新配置
    config_fold = config.copy()
    config_fold['output_dir'] = output_dir
    
    # 准备数据
    train_loader, val_loader, test_loader = prepare_data(config_fold, fold)
    
    # 创建TensorBoard日志记录器
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs",
        default_hp_metric=False
    )
    
    # 创建回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='{epoch:02d}-{val/dice:.4f}',
        monitor='val/dice',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val/dice',
        mode='max',
        patience=config.get('patience', 10),
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # 创建Trainer
    trainer = Trainer(
        max_epochs=config.get('max_epochs', 100),
        accelerator='gpu',
        devices=[int(config.get('gpu', 0))],
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        precision=config.get('precision', 16),
        gradient_clip_val=config.get('gradient_clip_val', 1.0)
    )
    
    # 创建模型
    model = BrainTumorSegTrainer(config_fold)
    
    # 训练模型
    trainer.fit(model, train_loader, val_loader)
    
    # 测试模型
    trainer.test(model, test_loader, ckpt_path='best')
    
    # 返回最佳模型路径和性能
    return {
        'fold': fold,
        'best_model_path': checkpoint_callback.best_model_path,
        'best_val_dice': model.best_val_dice,
        'val_dice_history': model.val_dice_history
    }

def train_all_folds(config, folds):
    """训练所有折"""
    results = []
    
    for fold in folds:
        print(f"\n{'='*20} 训练折 {fold} {'='*20}\n")
        result = train_model_fold(config, fold)
        results.append(result)
        print(f"\n{'='*20} 折 {fold} 训练完成, 最佳Dice: {result['best_val_dice']:.4f} {'='*20}\n")
    
    # 计算所有折的平均性能
    avg_dice = np.mean([r['best_val_dice'] for r in results])
    print(f"\n{'='*20} 所有折训练完成, 平均Dice: {avg_dice:.4f} {'='*20}\n")
    
    # 保存结果
    save_results(config, results, avg_dice)
    
    return results

def save_results(config, results, avg_dice):
    """保存训练结果"""
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果到文本文件
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write(f"模型类型: {config['model_type']}\n")
        f.write(f"平均Dice: {avg_dice:.4f}\n\n")
        
        for result in results:
            f.write(f"折 {result['fold']}:\n")
            f.write(f"  最佳Dice: {result['best_val_dice']:.4f}\n")
            f.write(f"  最佳模型路径: {result['best_model_path']}\n\n")

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    seed_everything(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config['gpu'] = args.gpu
    
    # 确定要训练的折
    if args.fold is not None:
        folds = [args.fold]
    else:
        folds = list(range(1, config.get('fold', 5) + 1))
    
    # 训练所有折
    results = train_all_folds(config, folds)
    
    print("训练完成!")

if __name__ == '__main__':
    main() 