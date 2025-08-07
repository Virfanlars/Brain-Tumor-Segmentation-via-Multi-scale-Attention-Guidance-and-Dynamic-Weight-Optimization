import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# 添加项目根目录到系统路径
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 修改为绝对导入
from models.network import BrainTumorSegNet, DynamicWeightLoss
from models.unet import UNet
from models.transunet import TransUNet
from models.hnfnetv2 import HNFNetv2

class BrainTumorSegTrainer(LightningModule):
    """脑肿瘤分割训练器"""
    
    def __init__(self, config):
        super(BrainTumorSegTrainer, self).__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        # 基础参数
        in_channels = config.get('in_channels', 1)
        num_classes = config.get('num_classes', 1)
        
        # 根据模型类型创建不同模型
        model_type = config.get('model_type', 'BrainTumorSegNet').lower()
        print(f"创建模型类型: {model_type}")
        
        if model_type == 'unet':
            self.model = UNet(
                in_channels=in_channels,
                out_channels=num_classes,
                features=config.get('features', [64, 128, 256, 512])
            )
            print("创建UNet模型完成")
            
        elif model_type == 'transunet':
            self.model = TransUNet(
                in_channels=in_channels,
                out_channels=num_classes,
                img_size=config.get('img_size', 256),
                patch_size=config.get('patch_size', 16),
                embed_dim=config.get('embed_dim', 768),
                depth=config.get('depth', 12),
                n_heads=config.get('n_heads', 12),
                features=config.get('features', [64, 128, 256, 512])
            )
            print("创建TransUNet模型完成")
            
        elif model_type == 'hnfnetv2':
            self.model = HNFNetv2(
                in_channels=in_channels,
                out_channels=num_classes,
                features=config.get('features', [64, 128, 256, 512])
            )
            print("创建HNF-Netv2模型完成")
            
        else:  # 默认使用BrainTumorSegNet
            self.model = BrainTumorSegNet(
                in_channels=in_channels,
                num_classes=num_classes,
                encoder_dims=config.get('encoder_dims', [64, 128, 256, 512]),
                decoder_dim=config.get('decoder_dim', 128),
                dropout_ratio=config.get('dropout_ratio', 0.1)
            )
            print("创建BrainTumorSegNet模型完成")
        
        # 创建损失函数
        initial_class_weights = config.get('class_weights', None)
        if initial_class_weights and isinstance(initial_class_weights, list):
            initial_class_weights = torch.tensor(initial_class_weights)
        
        self.criterion = DynamicWeightLoss(
            smooth=config.get('smooth', 1e-5),
            classes_weights=initial_class_weights,
            reduction='mean'
        )
        
        # 记录最佳性能
        self.best_val_dice = 0.0
        self.val_dice_history = []
        
        # 存储当前批次的指标
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 确保学习率是浮点数
        learning_rate = self.config.get('learning_rate', 3e-4)
        if isinstance(learning_rate, str):
            learning_rate = float(learning_rate)
            
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.999)
        )
        
        # 确保最小学习率是浮点数
        min_lr = self.config.get('min_lr', 1e-6)
        if isinstance(min_lr, str):
            min_lr = float(min_lr)
            
        # 余弦退火学习率调度
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.get('max_epochs', 100),
            eta_min=min_lr
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def training_step(self, batch, batch_idx):
        """单个训练步骤"""
        images = batch['image']
        masks = batch['mask']
        
        # 前向传播
        outputs = self.model(images)
        
        # 计算损失
        loss = self.criterion(outputs, masks)
        
        # 计算评估指标
        dice, iou, precision, recall = self.calculate_metrics(outputs, masks)
        
        # 记录指标
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/dice', dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/iou', iou, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train/precision', precision, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train/recall', recall, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        # 存储当前批次结果
        self.train_step_outputs.append({
            'loss': loss.detach(),
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall
        })
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """单个验证步骤"""
        images = batch['image']
        masks = batch['mask']
        
        # 前向传播
        outputs = self.model(images)
        
        # 计算损失
        loss = self.criterion(outputs, masks)
        
        # 计算评估指标
        dice, iou, precision, recall = self.calculate_metrics(outputs, masks)
        
        # 记录指标
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/dice', dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/iou', iou, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val/precision', precision, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val/recall', recall, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        # 存储当前批次结果
        self.val_step_outputs.append({
            'loss': loss.detach(),
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall
        })
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """单个测试步骤"""
        images = batch['image']
        masks = batch['mask']
        
        # 前向传播
        outputs = self.model(images)
        
        # 计算损失
        loss = self.criterion(outputs, masks)
        
        # 计算评估指标
        dice, iou, precision, recall = self.calculate_metrics(outputs, masks)
        
        # 存储当前批次结果
        self.test_step_outputs.append({
            'loss': loss.detach(),
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'pred': outputs.detach().cpu(),
            'mask': masks.detach().cpu(),
            'image': images.detach().cpu()
        })
        
        return loss
    
    def on_train_epoch_end(self):
        """训练轮结束回调"""
        # 计算平均指标
        avg_loss = torch.stack([x['loss'] for x in self.train_step_outputs]).mean()
        avg_dice = np.mean([x['dice'] for x in self.train_step_outputs])
        avg_iou = np.mean([x['iou'] for x in self.train_step_outputs])
        avg_precision = np.mean([x['precision'] for x in self.train_step_outputs])
        avg_recall = np.mean([x['recall'] for x in self.train_step_outputs])
        
        # 清空步骤输出
        self.train_step_outputs.clear()
        
        # 记录学习率
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train/lr', lr, on_step=False, on_epoch=True, logger=True)
    
    def on_validation_epoch_end(self):
        """验证轮结束回调"""
        # 计算平均指标
        avg_loss = torch.stack([x['loss'] for x in self.val_step_outputs]).mean()
        avg_dice = np.mean([x['dice'] for x in self.val_step_outputs])
        avg_iou = np.mean([x['iou'] for x in self.val_step_outputs])
        avg_precision = np.mean([x['precision'] for x in self.val_step_outputs])
        avg_recall = np.mean([x['recall'] for x in self.val_step_outputs])
        
        # 更新最佳性能
        if avg_dice > self.best_val_dice:
            self.best_val_dice = avg_dice
            self.log('val/best_dice', self.best_val_dice, on_step=False, on_epoch=True, logger=True)
        
        # 记录历史
        self.val_dice_history.append(avg_dice)
        
        # 清空步骤输出
        self.val_step_outputs.clear()
    
    def on_test_epoch_end(self):
        """测试轮结束回调"""
        # 计算平均指标
        avg_loss = torch.stack([x['loss'] for x in self.test_step_outputs]).mean()
        avg_dice = np.mean([x['dice'] for x in self.test_step_outputs])
        avg_iou = np.mean([x['iou'] for x in self.test_step_outputs])
        avg_precision = np.mean([x['precision'] for x in self.test_step_outputs])
        avg_recall = np.mean([x['recall'] for x in self.test_step_outputs])
        
        # 记录指标
        self.log('test/loss', avg_loss, on_step=False, on_epoch=True, logger=True)
        self.log('test/dice', avg_dice, on_step=False, on_epoch=True, logger=True)
        self.log('test/iou', avg_iou, on_step=False, on_epoch=True, logger=True)
        self.log('test/precision', avg_precision, on_step=False, on_epoch=True, logger=True)
        self.log('test/recall', avg_recall, on_step=False, on_epoch=True, logger=True)
        
        # 可视化部分结果
        if self.logger and len(self.test_step_outputs) > 0:
            # 选择第一个批次的前几个样本进行可视化
            num_samples = min(4, len(self.test_step_outputs[0]['image']))
            for i in range(num_samples):
                image = self.test_step_outputs[0]['image'][i, 0].numpy()
                mask = self.test_step_outputs[0]['mask'][i, 0].numpy()
                pred = torch.sigmoid(self.test_step_outputs[0]['pred'][i]).numpy()
                
                # 创建可视化图
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(image, cmap='gray')
                axes[0].set_title('Image')
                axes[0].axis('off')
                
                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                axes[2].imshow((pred > 0.5).astype(np.uint8).squeeze(), cmap='gray')
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                
                plt.tight_layout()
                
                # 添加到TensorBoard
                tensorboard = self.logger.experiment
                tensorboard.add_figure(f'test/sample_{i}', fig, self.current_epoch)
                plt.close(fig)
        
        # 清空步骤输出
        self.test_step_outputs.clear()
    
    def calculate_metrics(self, pred, target):
        """计算评估指标"""
        if pred.size(1) == 1:
            # 二分类问题
            pred_prob = torch.sigmoid(pred)
            pred_mask = (pred_prob > 0.5).float()
        else:
            # 多分类问题
            pred_prob = F.softmax(pred, dim=1)
            pred_mask = torch.argmax(pred_prob, dim=1, keepdim=True).float()
        
        # 确保target的形状正确
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # 将张量转移到CPU进行计算
        pred_mask = pred_mask.detach().cpu()
        target = target.detach().cpu()
        
        # 计算Dice系数
        intersection = (pred_mask * target).sum()
        union = pred_mask.sum() + target.sum()
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        
        # 计算IoU
        iou = (intersection + 1e-5) / (union - intersection + 1e-5)
        
        # 计算精确度和召回率
        true_positive = (pred_mask * target).sum()
        false_positive = pred_mask.sum() - true_positive
        false_negative = target.sum() - true_positive
        
        precision = true_positive / (true_positive + false_positive + 1e-5)
        recall = true_positive / (true_positive + false_negative + 1e-5)
        
        return dice.item(), iou.item(), precision.item(), recall.item()


def train_model(train_loader, val_loader, test_loader, config, fold=None):
    """训练模型的主函数"""
    # 创建保存目录
    output_dir = config.get('output_dir', 'output')
    
    # 为每个fold创建单独的子目录
    if fold is not None:
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
    else:
        fold_dir = output_dir
    
    # 创建TensorBoard日志记录器
    logger = TensorBoardLogger(
        save_dir=fold_dir,
        name="logs",
        default_hp_metric=False
    )
    
    # 创建回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(fold_dir, 'checkpoints'),
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
    
    # 创建Trainer - 对所有模型使用相同的训练设置
    trainer = Trainer(
        max_epochs=config.get('max_epochs', 100),
        accelerator='gpu',
        devices=[torch.cuda.current_device()],  # 使用当前设置的CUDA设备
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        precision=config.get('precision', 32),
        gradient_clip_val=config.get('gradient_clip_val', 1.0)
    )
    
    # 创建模型
    model = BrainTumorSegTrainer(config)
    
    # 训练模型
    trainer.fit(model, train_loader, val_loader)
    
    # 测试模型
    if test_loader:
        trainer.test(model, test_loader, ckpt_path='best')
    
    # 返回最佳模型路径和性能
    return {
        'best_model_path': checkpoint_callback.best_model_path,
        'best_val_dice': model.best_val_dice,
        'val_dice_history': model.val_dice_history
    } 