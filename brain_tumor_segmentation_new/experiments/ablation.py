import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import time

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from models.network import BrainTumorSegNet
from models.encoder import MixViTEncoder
from train.trainer import train_model
from data.dataset import create_dataloader
from evaluate.evaluate import calculate_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='脑肿瘤分割消融实验')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录，覆盖配置文件中的设置')
    parser.add_argument('--data_dir', type=str, default=None, help='数据目录，覆盖配置文件中的设置')
    parser.add_argument('--fold', type=int, default=1, help='指定要评估的折')
    parser.add_argument('--experiment', type=str, default='all', 
                        choices=['all', 'full', 'no_csda', 'no_dwd', 'fixed_dice', 'focal'],
                        help='指定要运行的消融实验，默认运行所有实验')
    
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
    
    if args.fold is not None:
        config['fold'] = args.fold
    
    return config

class BrainTumorSegNetWithoutCSDA(BrainTumorSegNet):
    """移除CSDA模块的脑肿瘤分割网络"""
    
    def __init__(self, in_channels=1, num_classes=1, encoder_dims=[64, 128, 256, 512],
                 decoder_dim=128, dropout_ratio=0.1):
        # 直接调用父类的 __init__ 方法而不是从 super() 中获取属性
        nn.Module.__init__(self)
        
        # 创建不使用CSDA的编码器
        self.encoder = MixViTEncoder(
            in_chans=in_channels,
            embed_dims=encoder_dims,
            depths=[2, 2, 6, 2],
            num_heads=[1, 2, 4, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            use_csda=False  # 关闭CSDA
        )
        
        # 创建解码器和分割头实例，不使用 super()
        from models.decoder import DynamicWeightDecoder, DynamicWeightHead
        
        # 解码器
        self.decoder = DynamicWeightDecoder(
            in_channels_list=encoder_dims,
            out_channels=decoder_dim,
            feature_strides=[4, 8, 16, 32]
        )
        
        # 分割头
        self.seg_head = DynamicWeightHead(
            in_channels=decoder_dim,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio
        )
        
        # 初始化权重
        self._init_weights()

class BrainTumorSegNetWithoutDWD(BrainTumorSegNet):
    """移除动态权重优化的脑肿瘤分割网络"""
    
    def __init__(self, in_channels=1, num_classes=1, encoder_dims=[64, 128, 256, 512],
                decoder_dim=128, dropout_ratio=0.1):
        super(BrainTumorSegNet, self).__init__()
        
        # 使用相同的编码器
        self.encoder = MixViTEncoder(
            in_chans=in_channels,
            embed_dims=encoder_dims,
            depths=[2, 2, 6, 2],
            num_heads=[1, 2, 4, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            use_csda=True
        )
        
        # 去除动态权重优化，使用标准解码器
        from models.decoder import ConvBNReLU
        
        # 创建简化的解码器
        self.decoder = nn.ModuleList()
        for i in range(len(encoder_dims)-1, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    ConvBNReLU(encoder_dims[i], decoder_dim),
                    nn.ConvTranspose2d(decoder_dim, decoder_dim, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(decoder_dim),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 最后一层解码器
        self.decoder.append(
            nn.Sequential(
                ConvBNReLU(encoder_dims[0], decoder_dim),
                nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True)
            )
        )
        
        # 分割头
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(decoder_dim // 2, num_classes, kernel_size=1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def forward(self, x):
        # 记录输入大小以便恢复
        input_size = x.size()[2:]
        
        # 编码器前向传播
        features = self.encoder(x)
        
        # 解码器前向传播 (不使用动态权重优化)
        x = features[-1]
        for i, decoder_block in enumerate(self.decoder[:-1]):
            x = decoder_block(x)
            # 添加跳跃连接
            if i < len(features) - 1:
                x = x + F.interpolate(features[-(i+2)], size=x.size()[2:], mode='bilinear', align_corners=False)
        
        # 最后一个解码块
        x = self.decoder[-1](x)
        
        # 调整大小到输入分辨率
        if x.size()[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        # 分割头
        x = self.seg_head(x)
        
        return x

def loss_factory(loss_type, **kwargs):
    """创建不同类型的损失函数"""
    if loss_type == 'dice':
        from models.network import DynamicWeightLoss
        return DynamicWeightLoss(**kwargs)
    
    elif loss_type == 'fixed_dice':
        from monai.losses import DiceLoss
        return DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    
    elif loss_type == 'focal':
        from torch.nn import BCEWithLogitsLoss
        class FocalLoss(nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.reduction = reduction
                self.bce = BCEWithLogitsLoss(reduction='none')
            
            def forward(self, pred, target):
                # 确保target的形状正确
                if target.dim() == 3:
                    target = target.unsqueeze(1)
                
                # 计算BCE损失
                bce_loss = self.bce(pred, target.float())
                
                # 计算Focal权重
                pt = torch.exp(-bce_loss)
                focal_weight = self.alpha * (1 - pt) ** self.gamma
                
                # 应用权重
                focal_loss = focal_weight * bce_loss
                
                # 应用reduction
                if self.reduction == 'mean':
                    return focal_loss.mean()
                elif self.reduction == 'sum':
                    return focal_loss.sum()
                else:
                    return focal_loss
        
        return FocalLoss(**kwargs)
    
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")

def run_ablation_experiment(config, model_type, loss_type=None):
    """运行单个消融实验"""
    # 创建数据集
    data_dir = config['data_dir']
    fold = config['fold']
    
    train_list = os.path.join(data_dir, f'fold_{fold}', 'train.txt')
    val_list = os.path.join(data_dir, f'fold_{fold}', 'val.txt')
    test_list = os.path.join(data_dir, f'fold_{fold}', 'test.txt')
    
    # 创建数据加载器
    train_loader = create_dataloader(
        train_list,
        batch_size=config.get('batch_size', 16),
        mode='train',
        crop_size=config.get('crop_size', (256, 256)),
        center_crop=config.get('center_crop', True),
        num_workers=config.get('num_workers', 4)
    )
    
    val_loader = create_dataloader(
        val_list,
        batch_size=config.get('batch_size', 16),
        mode='val',
        crop_size=config.get('crop_size', (256, 256)),
        center_crop=config.get('center_crop', True),
        num_workers=config.get('num_workers', 4)
    )
    
    test_loader = create_dataloader(
        test_list,
        batch_size=config.get('batch_size', 8),
        mode='test',
        crop_size=config.get('crop_size', (256, 256)),
        center_crop=config.get('center_crop', True),
        num_workers=config.get('num_workers', 4)
    )
    
    # 创建实验配置
    experiment_config = config.copy()
    experiment_config['exp_name'] = f"{config.get('exp_name', 'brain_tumor_seg')}_{model_type}"
    
    if loss_type:
        experiment_config['loss_type'] = loss_type
        experiment_config['exp_name'] += f"_{loss_type}"
    
    # 创建模型
    if model_type == 'full':
        # 完整模型
        model = BrainTumorSegNet(
            in_channels=config.get('in_channels', 1),
            num_classes=config.get('num_classes', 1),
            encoder_dims=config.get('encoder_dims', [64, 128, 256, 512]),
            decoder_dim=config.get('decoder_dim', 128),
            dropout_ratio=config.get('dropout_ratio', 0.1)
        )
    elif model_type == 'no_csda':
        # 不使用CSDA模块
        model = BrainTumorSegNetWithoutCSDA(
            in_channels=config.get('in_channels', 1),
            num_classes=config.get('num_classes', 1),
            encoder_dims=config.get('encoder_dims', [64, 128, 256, 512]),
            decoder_dim=config.get('decoder_dim', 128),
            dropout_ratio=config.get('dropout_ratio', 0.1)
        )
    elif model_type == 'no_dwd':
        # 不使用动态权重
        model = BrainTumorSegNetWithoutDWD(
            in_channels=config.get('in_channels', 1),
            num_classes=config.get('num_classes', 1),
            encoder_dims=config.get('encoder_dims', [64, 128, 256, 512]),
            decoder_dim=config.get('decoder_dim', 128),
            dropout_ratio=config.get('dropout_ratio', 0.1)
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 训练模型
    print(f"开始训练 {model_type} 模型...")
    result = train_model(train_loader, val_loader, test_loader, experiment_config, fold)
    
    return result

def run_all_ablation_experiments(config):
    """运行所有消融实验"""
    results = {}
    
    # 1. 完整模型 (基线)
    print("=" * 50)
    print("运行基线模型 (完整模型)...")
    results['full'] = run_ablation_experiment(config, 'full')
    
    # 2. 移除CSDA
    print("=" * 50)
    print("运行无CSDA模型...")
    results['no_csda'] = run_ablation_experiment(config, 'no_csda')
    
    # 3. 移除DWD
    print("=" * 50)
    print("运行无动态权重优化模型...")
    results['no_dwd'] = run_ablation_experiment(config, 'no_dwd')
    
    # 4. 损失函数 - 固定权重Dice
    print("=" * 50)
    print("运行固定权重Dice损失...")
    results['fixed_dice'] = run_ablation_experiment(config, 'full', 'fixed_dice')
    
    # 5. 损失函数 - Focal Loss
    print("=" * 50)
    print("运行Focal Loss...")
    results['focal'] = run_ablation_experiment(config, 'full', 'focal')
    
    return results

def plot_ablation_results(results, output_dir):
    """绘制消融实验结果"""
    # 创建表格数据
    data = {
        'Model': [],
        'Dice (ET)': [],
        'HD95 (mm)': [],
        'Parameters (M)': []
    }
    
    # 填充数据
    for model_type, result in results.items():
        data['Model'].append(model_type)
        data['Dice (ET)'].append(result.get('dice', 0))
        data['HD95 (mm)'].append(result.get('hd95', 0))
        data['Parameters (M)'].append(result.get('param_count', 0) / 1e6)
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存到CSV
    df.to_csv(os.path.join(output_dir, 'ablation_results.csv'), index=False)
    
    # 绘制Dice对比图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Model'], df['Dice (ET)'], color='skyblue')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient Comparison')
    plt.ylim(0.5, 1.0)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_comparison.png'), dpi=300)
    plt.close()
    
    # 绘制HD95对比图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Model'], df['HD95 (mm)'], color='salmon')
    plt.ylabel('HD95 (mm)')
    plt.title('Hausdorff Distance 95 Comparison')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hd95_comparison.png'), dpi=300)
    plt.close()
    
    # 绘制参数量对比图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Model'], df['Parameters (M)'], color='lightgreen')
    plt.ylabel('Parameters (M)')
    plt.title('Model Parameters Comparison')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameters_comparison.png'), dpi=300)
    plt.close()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    config = update_config(config, args)
    
    # 创建输出目录
    output_dir = os.path.join(config.get('output_dir', 'output'), 'ablation_studies')
    os.makedirs(output_dir, exist_ok=True)
    
    # 根据指定的实验类型运行
    start_time = time.time()
    
    if args.experiment == 'all':
        # 运行所有消融实验
        results = run_all_ablation_experiments(config)
    else:
        # 只运行指定的消融实验
        results = {}
        print("=" * 50)
        
        if args.experiment == 'full':
            print("运行基线模型 (完整模型)...")
            results['full'] = run_ablation_experiment(config, 'full')
        elif args.experiment == 'no_csda':
            print("运行无CSDA模型...")
            results['no_csda'] = run_ablation_experiment(config, 'no_csda')
        elif args.experiment == 'no_dwd':
            print("运行无动态权重优化模型...")
            results['no_dwd'] = run_ablation_experiment(config, 'no_dwd')
        elif args.experiment == 'fixed_dice':
            print("运行固定权重Dice损失...")
            results['fixed_dice'] = run_ablation_experiment(config, 'full', 'fixed_dice')
        elif args.experiment == 'focal':
            print("运行Focal Loss...")
            results['focal'] = run_ablation_experiment(config, 'full', 'focal')
    
    # 如果结果不为空,则绘制结果
    if results:
        plot_ablation_results(results, output_dir)
    
    # 计算总时间
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n消融实验完成！总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    print(f"结果保存在: {output_dir}")

if __name__ == "__main__":
    main() 