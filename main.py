import os
import sys
import argparse
import yaml
import time
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Brain Tumor Segmentation Project Main Entry')
    parser.add_argument('--mode', type=str, choices=['prepare', 'train', 'evaluate', 'visualize', 'all'],
                        required=True, help='Running mode')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Configuration file path')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input data directory for data preparation phase')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--format', type=str, default='npy', choices=['npy', 'jpg', 'png'],
                        help='Output format for data preparation phase')
    parser.add_argument('--fold', type=int, default=None,
                        help='Specify which fold to process')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Model path for evaluation and visualization')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--visualize_mode', type=str, default='overlay',
                        choices=['overlay', 'contour', 'difference', 'attention', '3d', 'all'],
                        help='Visualization mode')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Segmentation threshold for evaluation')
    
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def update_config(config, args):
    """Update configuration based on command line arguments"""
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    if args.fold is not None:
        config['fold'] = args.fold
    
    if args.model_path:
        config['model_path'] = args.model_path
        
    if args.resume:
        config['resume'] = args.resume
    
    return config

def prepare_data(args, config):
    """Data preparation phase"""
    from data.prepare_data import process_dataset
    
    input_dir = args.input_dir or 'dataset'
    output_dir = config.get('data_dir', 'processed_data')
    img_format = args.format
    n_splits = config.get('num_folds', 5)
    
    print(f"Starting data preparation: Processing data from {input_dir} to {output_dir}, format: {img_format}")
    process_dataset(input_dir, output_dir, img_format=img_format, n_splits=n_splits)
    
    return output_dir

def train_model(args, config):
    """Model training phase"""
    # Avoid circular import
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from train.train import main as train_main
    
    # Set command line arguments
    sys.argv = [
        sys.argv[0],
        '--config', args.config
    ]
    
    if args.fold is not None:
        sys.argv.extend(['--fold', str(args.fold)])
    
    if args.output_dir:
        sys.argv.extend(['--output_dir', args.output_dir])
        
    if args.resume:
        sys.argv.extend(['--resume', args.resume])
    
    print(f"Starting model training: Using configuration {args.config}")
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
    train_main()

def evaluate_model(args, config):
    """Model evaluation phase"""
    # Avoid circular import
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from evaluate.evaluate import main as evaluate_main
    
    # Set command line arguments
    sys.argv = [
        sys.argv[0],
        '--config', args.config,
        '--model_path', args.model_path or config.get('model_path', '')
    ]
    
    if args.fold is not None:
        sys.argv.extend(['--fold', str(args.fold)])
    
    if args.output_dir:
        sys.argv.extend(['--output_dir', args.output_dir])
    
    # 添加阈值参数
    threshold = args.threshold if hasattr(args, 'threshold') else 0.1
    sys.argv.extend(['--threshold', str(threshold)])
    
    # Enable visualization and save outputs
    sys.argv.extend(['--visualize', '--save_outputs'])
    
    print(f"Starting model evaluation: Using model {args.model_path or config.get('model_path', '')}")
    print(f"Segmentation threshold: {threshold}")
    evaluate_main()

def visualize_results(args, config):
    """Results visualization phase"""
    # Avoid circular import
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from visualize.visualize import main as visualize_main
    
    # Determine prediction results path
    if args.model_path:
        pred_path = os.path.join(config.get('output_dir', 'output'), 'predictions')
    else:
        pred_path = os.path.join(args.output_dir or config.get('output_dir', 'output'), 'predictions')
    
    # Determine output directory
    output_dir = os.path.join(args.output_dir or config.get('output_dir', 'output'), 'visualizations')
    
    # Set command line arguments
    sys.argv = [
        sys.argv[0],
        '--pred_path', pred_path,
        '--output_dir', output_dir,
        '--mode', args.visualize_mode
    ]
    
    print(f"Starting results visualization: Prediction results at {pred_path}, output to {output_dir}")
    visualize_main()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config = update_config(config, args)
    
    # Start timing
    start_time = time.time()
    
    # Execute operations based on mode
    if args.mode == 'prepare' or args.mode == 'all':
        data_dir = prepare_data(args, config)
        if args.mode == 'all':
            # Update data directory in configuration
            config['data_dir'] = data_dir
    
    if args.mode == 'train' or args.mode == 'all':
        train_model(args, config)
    
    if args.mode == 'evaluate' or args.mode == 'all':
        evaluate_model(args, config)
    
    if args.mode == 'visualize' or args.mode == 'all':
        visualize_results(args, config)
    
    # Calculate total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTotal running time: {int(hours)} hours {int(minutes)} minutes {seconds:.2f} seconds")
    print(f"Task '{args.mode}' completed!")

if __name__ == '__main__':
    main() 