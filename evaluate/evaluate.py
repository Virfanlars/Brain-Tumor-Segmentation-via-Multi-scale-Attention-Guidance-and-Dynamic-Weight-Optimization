import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
import medpy.metric.binary as medmetrics

# 注释掉中文字体设置
# plt.rcParams['font.sans-serif']=['SimHei'] # For normal display of Chinese labels
# plt.rcParams['axes.unicode_minus']=False # For normal display of minus sign

# 设置全局阈值
threshold = 0.1

# Add project root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from data.dataset import create_dataloader
from models.network import BrainTumorSegNet

def parse_args():
    parser = argparse.ArgumentParser(description='Brain Tumor Segmentation Evaluation Script')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--model_path', type=str, required=True, help='Model path')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory, overrides setting in config file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory, overrides setting in config file')
    parser.add_argument('--fold', type=int, default=None, help='Specify which fold to evaluate')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize results')
    parser.add_argument('--save_outputs', action='store_true', help='Whether to save prediction results')
    parser.add_argument('--threshold', type=float, default=0.1, help='Segmentation threshold (default: 0.1)')
    
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def update_config(config, args):
    """Update configuration based on command line arguments"""
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    if args.data_dir:
        config['data_dir'] = args.data_dir
    
    if args.model_path:
        config['model_path'] = args.model_path
    
    if args.fold is not None:
        config['fold'] = args.fold
    
    return config

def calculate_hausdorff(pred, target):
    """Calculate 95% Hausdorff distance"""
    try:
        return medmetrics.hd95(pred, target)
    except Exception as e:
        print(f"Warning: Error calculating Hausdorff95 distance - {e}")
        return float('nan')

def calculate_metrics(pred, target):
    """Calculate evaluation metrics"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # 调试输出，查看预测和目标的形状和值
    print(f"pred_shape: {pred.shape}")
    print(f"target_shape: {target.shape}")
    print(f"pred_unique: {np.unique(pred)}")
    print(f"target_unique: {np.unique(target)}")
    
    # 将二值化阈值从0.5降低到0.1
    threshold = 0.1
    
    # Ensure binary masks
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = (target > 0).astype(np.uint8)
    
    # 继续调试输出，查看二值化后的结果
    print(f"pred_binary_unique: {np.unique(pred_binary)}")
    print(f"target_binary_unique: {np.unique(target_binary)}")
    
    # Calculate Dice coefficient
    intersection = np.sum(pred_binary * target_binary)
    dice = (2.0 * intersection) / (np.sum(pred_binary) + np.sum(target_binary) + 1e-8)
    
    # Calculate IoU
    union = np.sum(pred_binary) + np.sum(target_binary) - intersection
    iou = intersection / (union + 1e-8)
    
    # Calculate Hausdorff distance
    if np.sum(pred_binary) > 0 and np.sum(target_binary) > 0:
        hd95 = calculate_hausdorff(pred_binary, target_binary)
    else:
        hd95 = float('nan')
    
    # Calculate precision and recall
    precision = np.sum(pred_binary * target_binary) / (np.sum(pred_binary) + 1e-8)
    recall = np.sum(pred_binary * target_binary) / (np.sum(target_binary) + 1e-8)
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'dice': dice,
        'iou': iou,
        'hd95': hd95,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_visualization(image, mask, pred, output_path, threshold=threshold):
    """Save visualization results"""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    
    # Ensure images are 2D
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]  # Convert from [1, H, W] to [H, W]
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    
    # Create heatmap
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Display original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Display mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Display prediction probability
    im = axes[2].imshow(pred, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title('Prediction Probability')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Display binarized prediction
    axes[3].imshow((pred > threshold).astype(np.uint8), cmap='gray')
    axes[3].set_title(f'Binarized Prediction (t={threshold})')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def evaluate_model(model, data_loader, config, args):
    """Evaluate model performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Prepare output directory
    output_dir = config.get('output_dir', 'output')
    vis_dir = os.path.join(output_dir, 'visualizations')
    if args.visualize:
        os.makedirs(vis_dir, exist_ok=True)
    
    pred_dir = os.path.join(output_dir, 'predictions')
    if args.save_outputs:
        os.makedirs(pred_dir, exist_ok=True)
    
    # Save all metrics
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            paths = batch['path']
            
            # 添加调试信息
            if batch_idx == 0:
                print(f"Input image shape: {images.shape}")
                print(f"Input image range: [{images.min().item():.4f}, {images.max().item():.4f}]")
                print(f"Label shape: {masks.shape}")
                print(f"Label range: [{masks.min().item():.4f}, {masks.max().item():.4f}]")
                print(f"Label unique values: {torch.unique(masks).cpu().numpy()}")
            
            # Inference
            outputs = model(images)
            
            # 添加更多调试信息
            if batch_idx == 0:
                print(f"Model output shape: {outputs.shape}")
                print(f"Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                print(f"Output first value: {outputs[0,0,0,0].item()}")
            
            # Apply sigmoid
            if outputs.size(1) == 1:
                probs = torch.sigmoid(outputs)
                if batch_idx == 0:
                    print(f"Using threshold {threshold} for single-channel output")
                    print(f"Prediction shape: {probs.shape}")
                    print(f"Prediction unique values: {torch.unique(probs > threshold).cpu().numpy()}")
                    # 查看第一个样本的预测结果
                    sample_pred = (probs[0] > threshold).cpu().numpy()
                    sample_mask = masks[0].cpu().numpy()
                    print(f"Binarized sample prediction unique values: {np.unique(sample_pred)}")
                    print(f"Sample label unique values: {np.unique(sample_mask)}")
            else:
                probs = F.softmax(outputs, dim=1)
                if batch_idx == 0:
                    print(f"Using argmax for multi-channel output")
                    
            # Calculate metrics for each sample
            for i in range(images.size(0)):
                image = images[i].cpu()
                mask = masks[i].cpu()
                prob = probs[i].cpu()
                path = paths[i]
                
                # Calculate metrics
                metrics = calculate_metrics(prob, mask)
                
                # Add filename
                file_name = os.path.basename(path)
                metrics['file_name'] = file_name
                all_metrics.append(metrics)
                
                # Visualize
                if args.visualize:
                    vis_path = os.path.join(vis_dir, f"{os.path.splitext(file_name)[0]}_viz.png")
                    save_visualization(image, mask, prob, vis_path, threshold=threshold)
                
                # Save prediction results
                if args.save_outputs:
                    pred_path = os.path.join(pred_dir, f"{os.path.splitext(file_name)[0]}_pred.npy")
                    np.save(pred_path, prob.numpy())
    
    # 添加类别分布统计
    class_counts = {}
    for metrics in all_metrics:
        pred_binary = np.load(os.path.join(pred_dir, f"{os.path.splitext(metrics['file_name'])[0]}_pred.npy")) > threshold
        for label in np.unique(pred_binary):
            class_counts[float(label)] = class_counts.get(float(label), 0) + pred_binary.size
    
    print("\nForeground pixel ratio: {:.4f}".format(class_counts.get(1.0, 0) / (class_counts.get(0.0, 0) + class_counts.get(1.0, 0))))
    print("Class distribution:", class_counts)
    
    # Calculate overall metrics
    metrics_df = pd.DataFrame(all_metrics)
    mean_metrics = metrics_df.mean(numeric_only=True)
    std_metrics = metrics_df.std(numeric_only=True)
    
    # Save detailed metrics to CSV
    metrics_file = os.path.join(output_dir, 'metrics_details.csv')
    metrics_df.to_csv(metrics_file, index=False)
    
    # Save summary metrics
    summary = pd.DataFrame({
        'Metric': mean_metrics.index,
        'Mean': mean_metrics.values,
        'Std': std_metrics.values
    })
    summary_file = os.path.join(output_dir, 'metrics_summary.csv')
    summary.to_csv(summary_file, index=False)
    
    # Output summary
    print("\nPerformance Summary:")
    for metric, value in mean_metrics.items():
        if metric not in ['file_name']:
            print(f"{metric}: {value:.4f} ± {std_metrics[metric]:.4f}")
    
    # Visualize metric distribution
    if args.visualize:
        plot_metrics_distribution(metrics_df, output_dir)
    
    return mean_metrics.to_dict()

def plot_metrics_distribution(metrics_df, output_dir):
    """Visualize metrics distribution"""
    metrics_to_plot = ['dice', 'iou', 'precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 4))
    
    for i, metric in enumerate(metrics_to_plot):
        axes[i].hist(metrics_df[metric].dropna(), bins=20, alpha=0.7, color='blue')
        axes[i].axvline(metrics_df[metric].mean(), color='red', linestyle='dashed', linewidth=2)
        axes[i].set_title(f'{metric.upper()} Distribution')
        axes[i].set_xlabel(metric)
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

def load_model(config, model_path):
    """Load model"""
    # Create model
    model = BrainTumorSegNet(
        in_channels=config.get('in_channels', 1),
        num_classes=config.get('num_classes', 1),
        encoder_dims=config.get('model', {}).get('encoder_dims', [64, 128, 256, 512]),
        decoder_dim=config.get('model', {}).get('decoder_dim', 128),
        dropout_ratio=config.get('model', {}).get('dropout_ratio', 0.1)
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    try:
        # Try to load state dict directly
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Failed to load model parameters directly: {e}")
        print("Using manual mapping mode to load parameters...")
        
        # Get model state dict and checkpoint state dict
        model_dict = model.state_dict()
        
        if 'state_dict' in checkpoint:
            checkpoint_dict = checkpoint['state_dict']
        else:
            checkpoint_dict = checkpoint
        
        # Create parameter mapping
        new_checkpoint_dict = {}
        
        # Manually create parameter name mapping rules
        param_mapping = {
            # encoder mapping
            'model.encoder.stages.': 'encoder.stages.',
            'model.encoder.norm.': 'encoder.norm.',
            
            # decoder mapping
            'model.decoder.lateral_convs.': 'decoder.lateral_convs.',
            'model.decoder.refine_modules.': 'decoder.refine_modules.',
            'model.decoder.fusion_modules.': 'decoder.fusion_modules.',
            'model.decoder.cross_attentions.': 'decoder.cross_attentions.',
            'model.decoder.seg_head.': 'decoder.seg_head.',
            
            # segmentation head mapping
            'model.seg_head.': 'seg_head.'
        }
        
        # Apply mapping rules
        for old_key, param in checkpoint_dict.items():
            if old_key.startswith('model.'):
                # Try to map parameter names
                new_key = None
                for old_prefix, new_prefix in param_mapping.items():
                    if old_key.startswith(old_prefix):
                        new_key = old_key.replace(old_prefix, new_prefix)
                        break
                
                if new_key and new_key in model_dict:
                    # Check if shapes match
                    if param.size() == model_dict[new_key].size():
                        new_checkpoint_dict[new_key] = param
                        print(f"Mapped parameter: {old_key} -> {new_key}")
            elif old_key in model_dict:
                # Direct match
                if param.size() == model_dict[old_key].size():
                    new_checkpoint_dict[old_key] = param
        
        # Update model dictionary
        model_dict.update(new_checkpoint_dict)
        
        # Load model
        model.load_state_dict(model_dict, strict=False)
        
        # Output loading information
        loaded_keys = set(new_checkpoint_dict.keys())
        model_keys = set(model_dict.keys())
        missing_keys = model_keys - loaded_keys
        
        print(f"Successfully loaded {len(loaded_keys)}/{len(model_keys)} parameters")
        print(f"There are {len(missing_keys)} parameters that could not be loaded, will use randomly initialized values")
    
    # Switch to evaluation mode
    model.eval()
    
    # Move to GPU (if available)
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config = update_config(config, args)
    
    # 更新全局阈值
    global threshold
    threshold = args.threshold
    print(f"Using segmentation threshold: {threshold}")
    
    # Ensure output directory exists
    output_dir = config.get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load model
    model_path = config.get('model_path')
    if not model_path:
        raise ValueError("Must provide model path")
    
    model = load_model(config, model_path)
    print(f"Loaded model from {model_path}")
    
    # Set up data loader
    data_dir = config.get('data_dir')
    fold = config.get('fold')
    
    if fold is not None:
        test_list = os.path.join(data_dir, f"fold_{fold}", "test.txt")
    else:
        test_list = os.path.join(data_dir, "test.txt")
    
    test_loader = create_dataloader(
        test_list,
        batch_size=config.get('batch_size', 8),
        mode='test',
        crop_size=config.get('crop_size', (256, 256)),
        center_crop=config.get('center_crop', True),
        num_workers=config.get('num_workers', 4)
    )
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, config, args)
    
    # Save configuration
    with open(os.path.join(output_dir, 'eval_config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    print(f"Evaluation completed. Detailed results saved in {output_dir}")
    
    return metrics

if __name__ == '__main__':
    main() 