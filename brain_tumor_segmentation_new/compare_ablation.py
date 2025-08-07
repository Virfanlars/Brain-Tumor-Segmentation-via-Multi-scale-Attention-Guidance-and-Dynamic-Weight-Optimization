import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
import torch.nn.functional as F
import yaml

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data.dataset import create_dataloader
from evaluate.evaluate import calculate_metrics as original_calculate_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Compare Ablation Study Models')
    parser.add_argument('--ckpt_paths', nargs='+', required=True,
                        help='List of model checkpoint paths')
    parser.add_argument('--model_names', nargs='+', default=None,
                        help='List of model names, should match the order of checkpoint paths')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                        help='Data directory')
    parser.add_argument('--fold', type=int, default=1,
                        help='Fold to evaluate')
    parser.add_argument('--output_dir', type=str, default='results/ablation_comparison',
                        help='Output directory')
    parser.add_argument('--visualize_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--config', type=str, default=None, help='模型结构配置yaml路径')
    
    return parser.parse_args()

def load_model(ckpt_path, model_type, config_path=None):
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint path does not exist: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        if model_type == 'fcn':
          from models.fcn import FCN8s
          model = FCN8s(num_classes=config.get('num_classes', 1))
        elif model_type == 'segnet':
          from models.segnet import SegNet
          model = SegNet(num_classes=config.get('num_classes', 1))
    # 去掉 'model.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict

    # 读取配置
    config = {}
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # 明确实例化模型
    if model_type == 'unet':
        from models.unet import UNet
        model = UNet(
            in_channels=config.get('in_channels', 1),
            out_channels=config.get('out_channels', 1),
            features=config.get('features', [64, 128, 256, 512])
        )
    elif model_type == 'transunet':
        from models.transunet import TransUNet
        model = TransUNet(
            in_channels=config.get('in_channels', 1),
            out_channels=config.get('out_channels', 1),
            img_size=config.get('img_size', 256),
            features=config.get('features', [64, 128, 256, 512]),
            patch_size=config.get('patch_size', 16),
            num_heads=config.get('num_heads', 4)
            # 其它TransUNet参数可继续补充
        )
    elif model_type == 'hnfnetv2':
        from models.hnfnetv2 import HNFNetv2
        model = HNFNetv2(
            in_channels=config.get('in_channels', 1),
            out_channels=config.get('out_channels', 1),
            features=config.get('features', [64, 128, 256, 512])
        )
    else:
        from models.network import BrainTumorSegNet
        model = BrainTumorSegNet(
            in_channels=config.get('in_channels', 1),
            num_classes=config.get('num_classes', 1),
            encoder_dims=config.get('encoder_dims', [64, 128, 256, 512]),
            decoder_dim=config.get('decoder_dim', 128)
        )
    model.load_state_dict(state_dict, strict=False)
    print("Model loaded with strict=False. 如果分割异常，请检查权重和模型类型、参数是否完全对应。")
    return model

def calculate_multi_class_metrics(pred, target):
    """Process multi-class segmentation metrics calculation"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Print shape information for debugging
    print(f"Prediction shape: {pred.shape}")
    print(f"Target shape: {target.shape}")
    
    # Handle multi-class segmentation
    if pred.shape[1] > 1:
        print(f"Detected multi-class segmentation, number of classes: {pred.shape[1]}")
        # Convert prediction to single channel (argmax)
        pred_argmax = np.argmax(pred, axis=1)
        
        # If target is also multi-class, convert to single channel
        if target.shape[1] > 1:
            target_argmax = np.argmax(target, axis=1)
        else:
            # If target is single channel, use directly
            target_argmax = target[:, 0]
        
        # Create binary mask for evaluation (foreground is any non-zero class)
        pred_binary = (pred_argmax > 0).astype(np.uint8)
        target_binary = (target_argmax > 0).astype(np.uint8)
        
        # Calculate metrics using binary mask
        return original_calculate_metrics(pred_binary, target_binary)
    else:
        # Single-class segmentation, use original function
        return original_calculate_metrics(pred, target)

def evaluate_model(model, data_loader, device):
    """Evaluate model performance"""
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []
    all_images = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Process batch data (now in dictionary format)
            images = batch['image'].to(device)
            targets = batch['mask'].to(device) if 'mask' in batch else None
            
            outputs = model(images)
            
            # Save predictions and targets for later metric calculation
            if outputs.shape[1] > 1:
                # Multi-class output, apply softmax
                preds = F.softmax(outputs, dim=1)
            else:
                # Binary segmentation, apply sigmoid
                preds = torch.sigmoid(outputs)
                
            all_preds.append(preds.cpu())
            if targets is not None:
                all_targets.append(targets.cpu())
            all_images.append(images.cpu())
    
    # Concatenate all batch predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0) if all_targets else None
    all_images = torch.cat(all_images, dim=0)
    
    # Calculate metrics
    metrics = calculate_multi_class_metrics(all_preds, all_targets)
    
    return metrics, all_preds, all_targets, all_images

def compare_models(ckpt_paths, model_names, data_loader, device, output_dir, config=None):
    """Compare performance of multiple models"""
    results = []
    all_predictions = []
    
    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"\nEvaluating model {i+1}/{len(ckpt_paths)}: {os.path.basename(ckpt_path)}")
        # Determine model name
        if model_names and i < len(model_names):
            model_name = model_names[i]
        else:
            model_name = f"Model-v{i+1}"
        
        # Load and evaluate model
        model = load_model(ckpt_path, None, config)
        metrics, predictions, targets, images = evaluate_model(model, data_loader, device)
        
        # Add results
        result = {"Model": model_name}
        result.update(metrics)
        results.append(result)
        
        # Save predictions for later visualization
        all_predictions.append(predictions)
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    return df, all_predictions, targets, images

def visualize_metrics(df, output_dir):
    """Visualize performance metrics"""
    # Set style
    sns.set(style="whitegrid")
    
    # Plot Dice and HD95 in separate figures for better spacing
    # Dice Coefficient plot
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x="Model", y="dice", data=df, palette="viridis")
    plt.title("Dice Coefficient Comparison", fontsize=16, pad=20)
    plt.ylabel("Dice Coefficient", fontsize=14)
    plt.ylim(0.4, 1.0)  # Adjust Y-axis for better visualization
    
    # Increase space for x-axis labels
    plt.subplots_adjust(bottom=0.3)
    
    # Rotate and align x-labels
    ax.set_xticklabels(ax.get_xticklabels(), bbox=None, rotation=20, ha='right', fontsize=12)
    
    # Display values on each bar
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.4f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', 
                   xytext = (0, 5), textcoords = 'offset points',
                   fontsize=12)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_dir, "dice_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # HD95 plot
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x="Model", y="hd95", data=df, palette="magma")
    plt.title("HD95 Distance Comparison", fontsize=16, pad=20)
    plt.ylabel("HD95 (mm)", fontsize=14)
    
    # Increase space for x-axis labels
    plt.subplots_adjust(bottom=0.3)
    
    # Rotate and align x-labels
    ax.set_xticklabels(ax.get_xticklabels(), bbox=None, rotation=20, ha='right', fontsize=12)
    
    # Display values on each bar
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', 
                   xytext = (0, 5), textcoords = 'offset points',
                   fontsize=12)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_dir, "hd95_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Combined metrics plot (for backward compatibility)
    plt.figure(figsize=(20, 10))
    
    # Plot Dice coefficient comparison
    plt.subplot(1, 2, 1)
    ax = sns.barplot(x="Model", y="dice", data=df, palette="viridis")
    plt.title("Dice Coefficient Comparison", fontsize=14, pad=15)
    plt.ylabel("Dice Coefficient", fontsize=12)
    plt.ylim(0.4, 1.0)
    
    ax.set_xticklabels(ax.get_xticklabels(), bbox=None, rotation=20, ha='right')
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.4f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', 
                   xytext = (0, 5), textcoords = 'offset points')
    
    # Plot HD95 comparison
    plt.subplot(1, 2, 2)
    ax = sns.barplot(x="Model", y="hd95", data=df, palette="magma")
    plt.title("HD95 Distance Comparison", fontsize=14, pad=15)
    plt.ylabel("HD95 (mm)", fontsize=12)
    
    ax.set_xticklabels(ax.get_xticklabels(), bbox=None, rotation=20, ha='right')
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', 
                   xytext = (0, 5), textcoords = 'offset points')
    
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.25)  # More space at the bottom for labels
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Create radar chart to compare all metrics
    metrics = [col for col in df.columns if col != "Model"]
    
    # Data preparation
    models = df["Model"].tolist()
    n_metrics = len(metrics)
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the plot
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    
    # Add axis labels for each metric with increased font size
    plt.xticks(angles[:-1], metrics, size=14)
    
    # Plot each model
    for i, model in enumerate(models):
        values = df.loc[i, metrics].values.tolist()
        # Normalize data (Dice: higher is better, others: lower is better)
        normalized_values = []
        for j, val in enumerate(values):
            if metrics[j] == "dice" or metrics[j] == "iou" or metrics[j] == "precision" or metrics[j] == "recall" or metrics[j] == "f1":
                # For metrics where higher is better
                max_val = df[metrics[j]].max()
                min_val = df[metrics[j]].min()
                if max_val == min_val:
                    normalized_values.append(1.0)  # Avoid division by zero
                else:
                    # Scale to [0.3, 1.0] range to ensure visibility
                    normalized_val = 0.3 + 0.7 * (val - min_val) / (max_val - min_val)
                    normalized_values.append(normalized_val)
            else:
                # For metrics where lower is better (hd95)
                max_val = df[metrics[j]].max()
                min_val = df[metrics[j]].min()
                if max_val == min_val:
                    normalized_values.append(0.3)  # Avoid division by zero
                else:
                    # Invert and scale to [0.3, 1.0] range
                    normalized_val = 0.3 + 0.7 * (max_val - val) / (max_val - min_val)
                    normalized_values.append(normalized_val)
        
        # Close the plot
        normalized_values += normalized_values[:1]
        ax.plot(angles, normalized_values, linewidth=2, label=model)
        ax.fill(angles, normalized_values, alpha=0.1)
    
    # Improve legend position and font size
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    plt.title("Model Performance Radar Chart", size=20, pad=20, bbox=None)
    plt.savefig(os.path.join(output_dir, "radar_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

def visualize_segmentations(all_predictions, targets, images, model_names, output_dir, num_samples=5):
    """Visualize segmentation results"""
    if not model_names:
        model_names = [f"Model-v{i+1}" for i in range(len(all_predictions))]
    
    # Limit number of samples
    num_samples = min(num_samples, images.shape[0])
    images = images[:num_samples]
    targets = targets[:num_samples] if targets is not None else None
    
    # Create visualizations for each sample
    os.makedirs(os.path.join(output_dir, "segmentation_samples"), exist_ok=True)
    
    for sample_idx in range(num_samples):
        plt.figure(figsize=(15, 10))
        
        # Display original image
        plt.subplot(2, len(model_names)+1, 1)
        plt.title("Original Image")
        plt.imshow(images[sample_idx, 0].numpy(), cmap='gray')
        plt.axis('off')
        
        # Display ground truth segmentation (if available)
        if targets is not None:
            plt.subplot(2, len(model_names)+1, len(model_names)+2)
            plt.title("Ground Truth")
            plt.imshow(images[sample_idx, 0].numpy(), cmap='gray')
            mask = targets[sample_idx, 0].numpy() > 0.5
            plt.imshow(mask, cmap='jet', alpha=0.5)
            plt.axis('off')
        
        # Display predictions from each model
        for i, preds in enumerate(all_predictions):
            pred = preds[sample_idx]
            
            # If multi-class segmentation, choose first class or combine all classes
            if pred.shape[0] > 1:
                # For multi-class, display argmax result
                pred_mask = pred.argmax(dim=0).numpy() > 0
            else:
                pred_mask = pred[0].numpy() > 0.5
            
            plt.subplot(2, len(model_names)+1, i+2)
            plt.title(model_names[i])
            plt.imshow(images[sample_idx, 0].numpy(), cmap='gray')
            plt.imshow(pred_mask, cmap='jet', alpha=0.5)
            plt.axis('off')
            
            # If ground truth is available, compute difference from ground truth
            if targets is not None:
                # Get ground truth mask
                if targets.shape[1] > 1:
                    true_mask = targets[sample_idx].argmax(dim=0).numpy() > 0
                else:
                    true_mask = targets[sample_idx, 0].numpy() > 0.5
                
                diff = pred_mask.astype(np.int8) - true_mask.astype(np.int8)
                
                plt.subplot(2, len(model_names)+1, i+len(model_names)+3)
                plt.title(f"{model_names[i]} vs GT")
                plt.imshow(images[sample_idx, 0].numpy(), cmap='gray')
                # Red for false positives, blue for false negatives
                plt.imshow(np.where(diff == 1, 1, 0), cmap='Reds', alpha=0.5)  # False Positive
                plt.imshow(np.where(diff == -1, 1, 0), cmap='Blues', alpha=0.5)  # False Negative
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "segmentation_samples", f"sample_{sample_idx+1}.png"), 
                    dpi=300, bbox_inches="tight")
        plt.close()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loader
    test_list = os.path.join(args.data_dir, f'fold_{args.fold}', 'test.txt')
    test_loader = create_dataloader(
        test_list,
        batch_size=8,
        mode='test',
        crop_size=(256, 256),
        center_crop=True,
        num_workers=4
    )
    
    # Ensure model names match number of checkpoints
    if args.model_names and len(args.model_names) != len(args.ckpt_paths):
        print(f"Warning: Number of model names ({len(args.model_names)}) does not match number of checkpoints ({len(args.ckpt_paths)})")
        args.model_names = None
    
    if not args.model_names:
        args.model_names = ["Full Model (Baseline)", "Model without CSDA", "Model without DWD", "Fixed Weight Model", "Focal Loss Model"]
        print(f"Using default model names: {args.model_names}")
    
    # Compare models
    df, all_predictions, targets, images = compare_models(
        args.ckpt_paths, 
        args.model_names, 
        test_loader, 
        device,
        args.output_dir,
        args.config
    )
    
    # Visualize metrics comparison
    visualize_metrics(df, args.output_dir)
    
    # Visualize segmentation results
    visualize_segmentations(
        all_predictions, 
        targets, 
        images, 
        args.model_names,
        args.output_dir,
        args.visualize_samples
    )
    
    print(f"\nComparison complete! Results saved to {args.output_dir}")
    print("\nPerformance metrics comparison:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main() 