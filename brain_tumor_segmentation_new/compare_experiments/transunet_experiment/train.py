#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TransUNet模型训练脚本
"""

import os
import sys

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)
sys.path.append(parent_dir)

# 使用模板中的训练函数
from train_template import main

if __name__ == '__main__':
    # 直接调用模板中的主函数
    main() 