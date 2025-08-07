import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CrossScaleAttention

class ConvBNReLU(nn.Module):
    """卷积+批归一化+ReLU块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FeatureRefinementModule(nn.Module):
    """特征细化模块，用于提取更细致的特征"""
    def __init__(self, in_channels, out_channels):
        super(FeatureRefinementModule, self).__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)
        self.atrous_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.atrous_bn1 = nn.BatchNorm2d(out_channels)
        self.atrous_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4, bias=False)
        self.atrous_bn2 = nn.BatchNorm2d(out_channels)
        self.atrous_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=8, dilation=8, bias=False)
        self.atrous_bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.project = ConvBNReLU(out_channels * 4, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        atrous1 = self.relu(self.atrous_bn1(self.atrous_conv1(x)))
        atrous2 = self.relu(self.atrous_bn2(self.atrous_conv2(x)))
        atrous3 = self.relu(self.atrous_bn3(self.atrous_conv3(x)))
        
        # 连接不同膨胀率的特征
        out = torch.cat([x, atrous1, atrous2, atrous3], dim=1)
        out = self.project(out)
        
        return out

class GatedFusion(nn.Module):
    """门控融合模块，根据特征重要性动态调整权重"""
    def __init__(self, high_channels, low_channels, out_channels):
        super(GatedFusion, self).__init__()
        self.conv_high = ConvBNReLU(high_channels, out_channels)
        self.conv_low = ConvBNReLU(low_channels, out_channels)
        
        # 门控计算网络
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, high_feat, low_feat):
        """
        高层特征: [B, high_channels, H1, W1]
        低层特征: [B, low_channels, H2, W2]
        """
        # 将高层特征上采样到低层特征的分辨率
        if high_feat.size(2) != low_feat.size(2) or high_feat.size(3) != low_feat.size(3):
            high_feat = F.interpolate(high_feat, size=low_feat.size()[2:], 
                                      mode='bilinear', align_corners=False)
        
        # 转换特征通道数
        high_feat = self.conv_high(high_feat)
        low_feat = self.conv_low(low_feat)
        
        # 计算门控权重
        concat_feat = torch.cat([high_feat, low_feat], dim=1)
        gate = self.gate(concat_feat)
        
        # 应用门控融合
        fused_feat = gate * high_feat + (1 - gate) * low_feat
        
        return fused_feat

class DynamicWeightDecoder(nn.Module):
    """动态权重解码器"""
    def __init__(self, in_channels_list, out_channels=64, feature_strides=[4, 8, 16, 32]):
        super(DynamicWeightDecoder, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.feature_strides = feature_strides
        
        # 特征转换模块
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(ConvBNReLU(in_channels, out_channels))
        
        # 特征细化模块
        self.refine_modules = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            self.refine_modules.append(FeatureRefinementModule(out_channels, out_channels))
        
        # 特征融合模块
        self.fusion_modules = nn.ModuleList()
        for i in range(len(in_channels_list) - 1):
            self.fusion_modules.append(GatedFusion(out_channels, out_channels, out_channels))
        
        # 跨尺度注意力模块
        self.cross_attentions = nn.ModuleList()
        for i in range(len(in_channels_list) - 1):
            self.cross_attentions.append(
                CrossScaleAttention(out_channels, out_channels, out_channels))
        
        # 最终输出头
        self.seg_head = nn.Sequential(
            ConvBNReLU(out_channels, out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, 1, kernel_size=1)
        )
    
    def forward(self, features):
        """
        features: 从编码器获取的多尺度特征列表
        """
        assert len(features) == len(self.in_channels_list)
        
        # 将所有特征转换为相同的通道数
        converted_features = []
        for i, feature in enumerate(features):
            feat = self.lateral_convs[i](feature)
            feat = self.refine_modules[i](feat)
            converted_features.append(feat)
        
        # 自顶向下的路径 - 从高语义层向低语义层传递
        # 从最深层开始，向浅层融合
        fused_features = [converted_features[-1]]  # 最深层的特征
        
        for i in range(len(converted_features) - 2, -1, -1):
            high_feat = fused_features[-1]
            low_feat = converted_features[i]
            
            # 使用跨尺度注意力进行特征融合
            cross_feat = self.cross_attentions[i](high_feat, low_feat)
            
            # 使用门控机制进一步融合
            fused_feat = self.fusion_modules[i](cross_feat, low_feat)
            
            fused_features.append(fused_feat)
        
        # 反转列表，使其从浅层到深层
        fused_features = fused_features[::-1]
        
        # 使用最浅层的特征进行分割预测
        out = self.seg_head(fused_features[0])
        
        return out, fused_features

class DynamicWeightHead(nn.Module):
    """动态权重分割头，支持多类别输出"""
    def __init__(self, in_channels, num_classes=1, dropout_ratio=0.1):
        super(DynamicWeightHead, self).__init__()
        self.num_classes = num_classes
        
        # 特征提取
        self.conv1 = ConvBNReLU(in_channels, in_channels)
        
        # 分割头
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        
        # 初始化权重
        nn.init.normal_(self.conv_seg.weight, mean=0, std=0.01)
        if self.conv_seg.bias is not None:
            nn.init.constant_(self.conv_seg.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        return x 