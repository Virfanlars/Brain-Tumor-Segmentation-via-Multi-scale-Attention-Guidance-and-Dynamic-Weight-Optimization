import os
import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 使用绝对路径导入
from data.dataset import create_dataloader
from train.trainer import train_model  # 修改为绝对导入

def parse_args():
    parser = argparse.ArgumentParser(description='脑肿瘤分割训练脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--fold', type=int, default=None, help='指定要训练的折，None表示训练所有折')
    parser.add_argument('--start_fold', type=int, default=1, help='指定从哪个折开始训练，默认从第1折开始')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录，覆盖配置文件中的设置')
    parser.add_argument('--data_dir', type=str, default=None, help='数据目录，覆盖配置文件中的设置')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--model_type', type=str, default=None, help='模型类型，例如unet, transunet, hnfnetv2, braintumsegnet')
    
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def update_config(config, args):
    """根据命令行参数更新配置"""
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    if args.data_dir:
        config['data_dir'] = args.data_dir
    
    if args.resume:
        config['resume'] = args.resume
    
    if args.model_type:
        config['model_type'] = args.model_type
        # 不再添加模型名称到输出路径，避免重复嵌套
        # 由脚本直接指定完整的模型输出路径
    
    return config

def train_fold(config, fold=None):
    """训练单个折"""
    print(f"{'=' * 20} 开始训练折 {fold} {'=' * 20}")
    
    # 更新配置
    fold_config = config.copy()
    fold_config['fold'] = fold
    
    # 确保输出目录存在
    output_dir = fold_config.get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"模型输出目录: {output_dir}")
    
    # 设置数据加载器
    data_dir = fold_config['data_dir']
    
    if fold is not None:
        train_list = os.path.join(data_dir, f'fold_{fold}', 'train.txt')
        val_list = os.path.join(data_dir, f'fold_{fold}', 'val.txt')
        test_list = os.path.join(data_dir, f'fold_{fold}', 'test.txt')
    else:
        train_list = os.path.join(data_dir, 'train.txt')
        val_list = os.path.join(data_dir, 'val.txt')
        test_list = os.path.join(data_dir, 'test.txt')
    
    # 创建数据加载器
    train_loader = create_dataloader(
        train_list,
        batch_size=fold_config.get('batch_size', 16),
        mode='train',
        crop_size=fold_config.get('crop_size', (256, 256)),
        center_crop=fold_config.get('center_crop', True),
        num_workers=fold_config.get('num_workers', 4)
    )
    
    val_loader = create_dataloader(
        val_list,
        batch_size=fold_config.get('batch_size', 16),
        mode='val',
        crop_size=fold_config.get('crop_size', (256, 256)),
        center_crop=fold_config.get('center_crop', True),
        num_workers=fold_config.get('num_workers', 4)
    )
    
    test_loader = create_dataloader(
        test_list,
        batch_size=fold_config.get('batch_size', 8),
        mode='test',
        crop_size=fold_config.get('crop_size', (256, 256)),
        center_crop=fold_config.get('center_crop', True),
        num_workers=fold_config.get('num_workers', 4)
    )
    
    # 训练模型
    result = train_model(train_loader, val_loader, test_loader, fold_config, fold)
    
    print(f"折 {fold} 训练完成，最佳验证Dice系数: {result['best_val_dice']:.4f}")
    print(f"最佳模型保存在: {result['best_model_path']}")
    
    return result

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    config = update_config(config, args)
    
    # 确保输出目录存在
    output_dir = config.get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # 确保所有模型都执行完整的五折交叉验证
    if args.fold is not None:
        # 只训练指定的折
        train_fold(config, args.fold)
    else:
        # 确保训练所有五折
        num_folds = config.get('num_folds', 5)
        results = []
        
        # 使用start_fold参数决定从哪个折开始训练
        for fold in range(args.start_fold, num_folds + 1):
            result = train_fold(config, fold)
            results.append(result)
        
        # 输出所有折的性能
        best_dice_scores = [result['best_val_dice'] for result in results]
        mean_dice = np.mean(best_dice_scores)
        std_dice = np.std(best_dice_scores)
        
        print(f"\n{'=' * 50}")
        print(f"模型 {config.get('model_type', 'default')} 的交叉验证结果 ({len(results)} 折):")
        print(f"平均Dice系数: {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"各折Dice系数: {[f'{dice:.4f}' for dice in best_dice_scores]}")
        print(f"{'=' * 50}")

if __name__ == '__main__':
    main() 