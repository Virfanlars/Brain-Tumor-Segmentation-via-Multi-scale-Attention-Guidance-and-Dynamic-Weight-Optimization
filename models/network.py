import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import MixViTEncoder
from .decoder import DynamicWeightDecoder, DynamicWeightHead

class BrainTumorSegNet(nn.Module):
    """基于多尺度注意力引导与动态权重优化的脑肿瘤分割网络"""
    
    def __init__(self, in_channels=1, num_classes=1, encoder_dims=[64, 128, 256, 512],
                 decoder_dim=128, dropout_ratio=0.1):
        super(BrainTumorSegNet, self).__init__()
        
        # 编码器
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
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        """前向传播
        
        参数:
            x (Tensor): 输入图像 [B, C, H, W]
            return_features (bool): 是否返回中间特征
            
        返回:
            out (Tensor): 分割预测 [B, num_classes, H, W]
            features (list): 可选，中间特征
        """
        # 编码器前向传播
        encoder_features = self.encoder(x)
        
        # 解码器前向传播
        decoder_output, decoder_features = self.decoder(encoder_features)
        
        # 调整大小到输入分辨率
        if decoder_output.size()[2:] != x.size()[2:]:
            decoder_output = F.interpolate(
                decoder_output, 
                size=x.size()[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # 如果需要返回特征
        if return_features:
            return decoder_output, {"encoder_features": encoder_features, "decoder_features": decoder_features}
        
        return decoder_output

class DynamicWeightLoss(nn.Module):
    """动态权重损失函数，结合Dice损失和交叉熵损失"""
    
    def __init__(self, smooth=1e-5, classes_weights=None, reduction='mean', ignore_index=-100):
        super(DynamicWeightLoss, self).__init__()
        self.smooth = smooth
        self.classes_weights = classes_weights
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        # 使用BCE Loss作为辅助损失
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, pred, target, weights=None):
        """计算损失
        
        参数:
            pred (Tensor): 预测张量 [B, C, H, W]
            target (Tensor): 目标张量 [B, H, W] 或 [B, C, H, W]
            weights (Tensor, optional): 样本权重 [B]
            
        返回:
            loss (Tensor): 损失值
        """
        # 确保target的形状正确 [B, H, W] -> [B, 1, H, W]
        if target.dim() == 3:
            target = target.unsqueeze(1)
            
        # 确保target的维度顺序与pred一致
        if target.shape != pred.shape and target.shape[-3:] == pred.shape[-3:]:
            # 如果通道在最后一维，需要调整
            if target.shape[-1] == 1:
                target = target.permute(0, 3, 1, 2)
        
        # 如果是多类别，就转换为one-hot编码
        if pred.size(1) > 1 and target.size(1) == 1:
            # 将目标从 [B, 1, H, W] 转换为 [B, C, H, W]
            target_one_hot = torch.zeros_like(pred)
            target_one_hot.scatter_(1, target.long(), 1)
            target = target_one_hot
        
        # 应用sigmoid/softmax
        if pred.size(1) == 1:
            # 二分类问题
            pred_prob = torch.sigmoid(pred)
        else:
            # 多分类问题
            pred_prob = F.softmax(pred, dim=1)
        
        # 计算Dice系数
        batch_size = pred.size(0)
        num_classes = pred.size(1)
        
        # 计算每个类别的Dice损失
        dice_losses = []
        for c in range(num_classes):
            pred_c = pred_prob[:, c]
            target_c = target[:, c]
            
            # 计算交集和并集
            intersection = (pred_c * target_c).sum(dim=(1, 2))
            pred_sum = pred_c.sum(dim=(1, 2))
            target_sum = target_c.sum(dim=(1, 2))
            
            # 计算Dice系数
            dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            
            # 转换为损失
            dice_loss = 1.0 - dice
            
            # 处理权重
            if weights is not None:
                dice_loss = dice_loss * weights
            
            # 应用类别权重
            if self.classes_weights is not None:
                dice_loss = dice_loss * self.classes_weights[c]
            
            dice_losses.append(dice_loss)
        
        # 合并各类别的损失
        dice_loss = torch.stack(dice_losses, dim=1)
        
        # 计算BCE损失
        if pred.size(1) == 1:
            # 二分类问题
            bce_loss = self.bce_loss(pred, target.float())
            bce_loss = bce_loss.mean(dim=(1, 2, 3))
        else:
            # 多分类问题
            bce_loss = F.cross_entropy(
                pred, target.argmax(dim=1), 
                reduction='none', 
                ignore_index=self.ignore_index
            )
            bce_loss = bce_loss.mean(dim=(1, 2))
        
        # 根据损失大小动态调整权重
        # 如果某个类别的dice损失更大，应该给予更多的权重
        total_loss = dice_loss.mean(dim=1) * 0.7 + bce_loss * 0.3
        
        # 应用reduction
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss

def create_model(config):
    """根据配置创建模型"""
    model = BrainTumorSegNet(
        in_channels=config.get('in_channels', 1),
        num_classes=config.get('num_classes', 1),
        encoder_dims=config.get('encoder_dims', [64, 128, 256, 512]),
        decoder_dim=config.get('decoder_dim', 128),
        dropout_ratio=config.get('dropout_ratio', 0.1)
    )
    return model 