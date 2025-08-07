import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "空间注意力的卷积核大小必须为3或7"
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 沿通道维度计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 连接特征图并应用卷积
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        
        return self.sigmoid(x)

class CSDA(nn.Module):
    """增强版通道-空间双重注意力模块，集成跨尺度注意力"""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CSDA, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
        # 添加多尺度特征提取
        self.scales = [1, 0.5, 0.25]  # 多个尺度
        self.multi_scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ) for _ in self.scales
        ])
        
        # 跨尺度注意力
        self.cross_scale_attention = CrossScaleAttention(
            high_dim=in_channels,
            low_dim=in_channels,
            out_dim=in_channels,
            reduction=4
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 1. 基础的通道-空间注意力
        cs_att = x * self.channel_attention(x)
        cs_att = cs_att * self.spatial_attention(cs_att)
        
        # 2. 多尺度特征提取
        multi_scale_feats = []
        for i, scale in enumerate(self.scales):
            if scale != 1:
                # 下采样
                scaled_x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
                # 特征提取
                scaled_feat = self.multi_scale_convs[i](scaled_x)
                # 上采样回原始大小
                scaled_feat = F.interpolate(scaled_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
            else:
                scaled_feat = self.multi_scale_convs[i](x)
            multi_scale_feats.append(scaled_feat)
        
        # 3. 融合多尺度特征
        multi_scale_feat = sum(multi_scale_feats) / len(multi_scale_feats)
        
        # 4. 应用跨尺度注意力
        # 使用原始特征作为high_feat，多尺度融合特征作为low_feat
        cross_scale_feat = self.cross_scale_attention(cs_att, multi_scale_feat)
        
        # 5. 特征融合
        output = self.fusion(torch.cat([cs_att, cross_scale_feat], dim=1))
        
        # 6. 残差连接
        output = output + x
        
        return output

class MultiheadSelfAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"维度 {dim} 必须能被头数 {num_heads} 整除."
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LinearAttention(nn.Module):
    """线性注意力机制，降低计算复杂度"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.):
        super().__init__()
        assert dim % num_heads == 0, "维度必须能被头数整除"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.elu = nn.ELU()
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 使用ELU+1确保k是正的
        k = self.elu(k) + 1.0
        
        # 线性注意力
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        
        # 线性上下文描述
        context = k.transpose(-2, -1) @ v  # [B, num_heads, head_dim, head_dim]
        out = (q @ context).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class CrossScaleAttention(nn.Module):
    """跨尺度注意力模块，融合不同尺度特征"""
    def __init__(self, high_dim, low_dim, out_dim, reduction=4):
        super(CrossScaleAttention, self).__init__()
        self.high_channels = high_dim
        self.low_channels = low_dim
        
        # 对高层特征降维
        self.high_conv = nn.Sequential(
            nn.Conv2d(high_dim, out_dim // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim // reduction),
            nn.ReLU(inplace=True)
        )
        
        # 对低层特征降维
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_dim, out_dim // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim // reduction),
            nn.ReLU(inplace=True)
        )
        
        # 注意力计算
        self.query_conv = nn.Conv2d(out_dim // reduction, out_dim // reduction, kernel_size=1)
        self.key_conv = nn.Conv2d(out_dim // reduction, out_dim // reduction, kernel_size=1)
        self.value_high_conv = nn.Conv2d(high_dim, out_dim, kernel_size=1)
        self.value_low_conv = nn.Conv2d(low_dim, out_dim, kernel_size=1)
        
        # 输出投影
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, high_feat, low_feat):
        """
        高层特征: [B, high_dim, H1, W1]
        低层特征: [B, low_dim, H2, W2]
        """
        batch_size = high_feat.size(0)
        
        # 将低层特征上采样到高层特征的大小
        if high_feat.size(2) != low_feat.size(2) or high_feat.size(3) != low_feat.size(3):
            low_feat = F.interpolate(low_feat, size=high_feat.size()[2:], mode='bilinear', align_corners=False)
        
        # 特征降维
        high_feat_embed = self.high_conv(high_feat)
        low_feat_embed = self.low_conv(low_feat)
        
        # 计算query和key
        query = self.query_conv(high_feat_embed).view(batch_size, -1, high_feat_embed.size(2) * high_feat_embed.size(3))
        key = self.key_conv(low_feat_embed).view(batch_size, -1, low_feat_embed.size(2) * low_feat_embed.size(3))
        
        # 计算注意力图
        energy = torch.bmm(query.permute(0, 2, 1), key)  # [B, H1*W1, H2*W2]
        attention = self.softmax(energy)  # 对每个像素点的所有通道进行softmax
        
        # 计算高层和低层的value
        value_high = self.value_high_conv(high_feat).view(batch_size, -1, high_feat.size(2) * high_feat.size(3))
        value_low = self.value_low_conv(low_feat).view(batch_size, -1, low_feat.size(2) * low_feat.size(3))
        
        # 应用注意力
        out_high = torch.bmm(value_high, attention.permute(0, 2, 1))
        out_low = torch.bmm(value_low, attention)
        
        # 重塑输出
        out_high = out_high.view(batch_size, -1, high_feat.size(2), high_feat.size(3))
        out_low = out_low.view(batch_size, -1, low_feat.size(2), low_feat.size(3))
        
        # 连接并投影
        out = torch.cat([out_high, out_low], dim=1)
        out = self.out_conv(out)
        
        # 残差连接
        out = high_feat + self.gamma * out
        
        return out 