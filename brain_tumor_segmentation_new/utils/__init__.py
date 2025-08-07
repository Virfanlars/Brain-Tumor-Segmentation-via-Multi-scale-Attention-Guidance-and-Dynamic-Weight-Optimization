# 通用工具函数

import os
import numpy as np
import torch
import random
import logging
import yaml
from pathlib import Path

def set_seed(seed=42):
    """设置随机种子，确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_dir, name='train', level=logging.INFO):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志对象
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 文件处理器
    file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
    file_handler.setLevel(level)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_yaml(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_yaml(config, save_path):
    """保存YAML配置文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)

def calculate_model_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

__all__ = [
    'set_seed',
    'setup_logging',
    'load_yaml',
    'save_yaml',
    'calculate_model_parameters'
] 