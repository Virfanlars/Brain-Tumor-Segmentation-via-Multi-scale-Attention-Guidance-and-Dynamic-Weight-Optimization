# 脑肿瘤分割实验指南

本目录包含用于评估脑肿瘤分割方法的实验脚本。

## 实验目录结构

```
experiments/
├── ablation.py           # 消融实验脚本(评估不同组件的贡献)
├── ablation_study.py     # 消融实验脚本(简化版)
├── comparison.py         # 对比实验脚本(与其他方法比较)
└── README.md             # 本说明文档
```

## 环境准备

在运行实验前，确保已安装所有依赖：

```bash
pip install -r ../requirements.txt
```

## 模型准备

1. 确保已训练好所需的模型权重，并将其放在正确的位置：

```
brain_tumor_segmentation/weights/
├── braintumorsegnet.pth   # 完整的提出方法模型权重
├── no_csda.pth            # 无CSDA模块的模型权重
├── no_dwd.pth             # 无动态权重解码器的模型权重
├── unet.pth               # UNet模型权重
├── transunet.pth          # TransUNet模型权重
└── hnfnetv2.pth           # HNF-Netv2模型权重
```

2. 如果尚未训练模型，可以使用以下命令训练：

```bash
# 训练完整模型
python train/train.py --config configs/train_config.yaml

# 训练消融实验模型
python train/train.py --config configs/ablation_config.yaml --model_type no_csda
python train/train.py --config configs/ablation_config.yaml --model_type no_dwd
python train/train.py --config configs/ablation_config.yaml --model_type full --loss_type fixed_dice
```

## 运行对比实验

对比实验将提出的方法与现有的SOTA方法进行比较，包括UNet、TransUNet和HNF-Netv2。

### 准备工作

1. 确保已经处理好数据，数据应位于`../processed_data`目录
2. 确保已经训练好模型权重，或将预训练权重放置在正确位置：

```bash
# 创建权重目录（如果不存在）
mkdir -p ../weights

# 如果使用相对路径访问数据，创建符号链接
ln -s ../processed_data ./processed_data
```

### 运行实验

```bash
# 使用相对路径（推荐，需要先创建符号链接）
python comparison.py --config ../configs/compare_config.yaml --data_dir ./processed_data --visualize

# 或使用绝对路径
python comparison.py --config ../configs/compare_config.yaml --data_dir /home/wf/data1/cut/brain_tumor_segmentation/processed_data --visualize
```

如果出现找不到模型权重的警告，请确保模型权重文件存在于`../weights/`目录下：
- unet.pth
- transunet.pth
- hnfnetv2.pth
- braintumorsegnet.pth

参数说明：
- `--config`: 配置文件路径，默认为`../configs/compare_config.yaml`
- `--data_dir`: 数据目录，覆盖配置文件中的设置
- `--output_dir`: 结果输出目录，覆盖配置文件中的设置
- `--visualize`: 是否生成可视化结果，默认根据配置文件设置

结果将保存在`results/comparison`目录中（可通过配置文件或命令行参数更改）：
- `comparison_results.csv`: 性能指标结果表
- `metrics_comparison.png`: 性能指标条形图
- `inference_time_comparison.png`: 推理时间条形图
- `metrics_heatmap.png`: 性能指标热力图
- `visualizations/`: 分割结果可视化目录

## 运行消融实验

消融实验用于评估不同组件对模型性能的贡献，包括CSDA模块、动态权重解码器和不同的损失函数。

```bash
python ablation.py --config ../configs/ablation_config.yaml --output_dir results/ablation
```

参数说明：
- `--config`: 配置文件路径，默认为`../configs/ablation_config.yaml`
- `--data_dir`: 数据目录，覆盖配置文件中的设置
- `--output_dir`: 结果输出目录，覆盖配置文件中的设置
- `--fold`: 指定要评估的折，默认为1

结果将保存在`results/ablation`目录中（可通过配置文件或命令行参数更改）：
- `ablation_results.csv`: 性能指标结果表
- `dice_comparison.png`: Dice系数比较图
- `hd95_comparison.png`: HD95距离比较图
- `parameters_comparison.png`: 模型参数量比较图

## 配置文件说明

### 对比实验配置

配置文件(`configs/compare_config.yaml`)示例：

```yaml
# 数据设置
data_dir: "data/processed"  # 数据目录
fold: 1  # 使用的交叉验证折
img_size: [256, 256]  # 输入图像大小
batch_size: 8  # 批量大小

# 模型设置
models:
  unet:
    name: "UNet"
    weight_path: "weights/unet.pth"
    features: [64, 128, 256, 512]
  
  transunet:
    name: "TransUNet"
    weight_path: "weights/transunet.pth"
    ...

# 评估设置
eval:
  metrics: ["dice", "iou", "precision", "recall", "specificity", "hd95"]
  visualize: true  # 是否生成可视化结果
```

### 消融实验配置

配置文件(`configs/ablation_config.yaml`)示例：

```yaml
# 数据设置
data_dir: "data/processed"  # 数据目录
fold: 1  # 使用的交叉验证折

# 消融设置
ablation:
  experiments:
    - name: "完整模型"
      model_type: "full"
      use_csda: true
      use_dwd: true
      loss_type: "dynamic"
      
    - name: "无CSDA"
      model_type: "no_csda"
      use_csda: false
      ...
```

## 注意事项

1. 确保数据集按照正确的格式准备，包括交叉验证折的划分。
2. 可视化功能会消耗较多内存，如果遇到内存问题，请减少可视化样本数量。
3. 在大型数据集上运行实验前，建议先使用小数据集测试脚本功能。
4. 如果遇到CUDA内存不足问题，请减小批量大小(`batch_size`)。 