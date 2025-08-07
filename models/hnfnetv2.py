import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class NestedBlock(nn.Module):
    """嵌套式块，用于HNF-Netv2中的特征提取"""
    def __init__(self, in_channels, mid_channels, out_channels):
        super(NestedBlock, self).__init__()
        self.conv1 = ConvBNReLU(in_channels, mid_channels)
        self.conv2 = ConvBNReLU(mid_channels, out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class AttentionGate(nn.Module):
    """注意力门控机制"""
    def __init__(self, f_channels, g_channels, int_channels):
        super(AttentionGate, self).__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(g_channels, int_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(int_channels)
        )
        self.Wf = nn.Sequential(
            nn.Conv2d(f_channels, int_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(int_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(int_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, f, g):
        # f: 低级特征, g: 高级特征(门控信号)
        g_temp = self.Wg(g)
        f_temp = self.Wf(f)
        
        if f_temp.shape[2:] != g_temp.shape[2:]:
            g_temp = F.interpolate(g_temp, size=f_temp.shape[2:], mode='bilinear', align_corners=True)
            
        result = self.relu(f_temp + g_temp)
        att_map = self.psi(result)
        
        return f * att_map

class FeatureFusion(nn.Module):
    """特征融合模块"""
    def __init__(self, channels):
        super(FeatureFusion, self).__init__()
        self.conv = ConvBNReLU(channels*2, channels)
        
    def forward(self, x1, x2):
        # 确保x1和x2的形状一致
        if x1.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=True)
            
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class MultiScaleFeatureAggregation(nn.Module):
    """多尺度特征聚合模块"""
    def __init__(self, channels_list):
        super(MultiScaleFeatureAggregation, self).__init__()
        
        # 转换每个尺度的特征到相同通道数
        self.adapters = nn.ModuleList([
            ConvBNReLU(ch, channels_list[-1]) for ch in channels_list
        ])
        
        # 最终融合
        self.fusion = ConvBNReLU(channels_list[-1] * len(channels_list), channels_list[-1])
        
    def forward(self, features):
        # 转换并对齐特征大小
        target_size = features[0].shape[2:]  # 使用第一个特征的大小作为目标大小
        
        aligned_features = []
        for i, feature in enumerate(features):
            feature = self.adapters[i](feature)
            if feature.shape[2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode='bilinear', align_corners=True)
            aligned_features.append(feature)
        
        # 合并特征
        fused = torch.cat(aligned_features, dim=1)
        return self.fusion(fused)

class HNFNetv2(nn.Module):
    """层级嵌套特征网络v2，用于医学图像分割"""
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(HNFNetv2, self).__init__()
        
        # 编码器部分
        self.encoder1 = NestedBlock(in_channels, features[0]//2, features[0])
        self.encoder2 = NestedBlock(features[0], features[1]//2, features[1])
        self.encoder3 = NestedBlock(features[1], features[2]//2, features[2])
        self.encoder4 = NestedBlock(features[2], features[3]//2, features[3])
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 解码器部分
        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        
        # 注意力门
        self.att3 = AttentionGate(features[2], features[3], features[2]//2)
        self.att2 = AttentionGate(features[1], features[2], features[1]//2)
        self.att1 = AttentionGate(features[0], features[1], features[0]//2)
        
        # 特征融合
        self.fusion3 = FeatureFusion(features[2])
        self.fusion2 = FeatureFusion(features[1])
        self.fusion1 = FeatureFusion(features[0])
        
        # 嵌套连接解码器
        self.decoder3 = NestedBlock(features[2]*2, features[2], features[2])
        self.decoder2 = NestedBlock(features[1]*2, features[1], features[1])
        self.decoder1 = NestedBlock(features[0]*2, features[0], features[0])
        
        # 多尺度特征聚合
        self.msfa = MultiScaleFeatureAggregation([features[0], features[1], features[2], features[3]])
        
        # 输出层
        self.final = nn.Conv2d(features[-1], out_channels, kernel_size=1)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
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
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # 记录原始大小
        input_size = x.size()[2:]
        
        # 编码器路径
        e1 = self.encoder1(x)
        e1_pool = self.pool(e1)
        
        e2 = self.encoder2(e1_pool)
        e2_pool = self.pool(e2)
        
        e3 = self.encoder3(e2_pool)
        e3_pool = self.pool(e3)
        
        e4 = self.encoder4(e3_pool)
        
        # 解码器路径与注意力门控
        d3 = self.upconv3(e4)
        a3 = self.att3(e3, e4)
        d3 = self.fusion3(d3, a3)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))
        
        d2 = self.upconv2(d3)
        a2 = self.att2(e2, d3)
        d2 = self.fusion2(d2, a2)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))
        
        d1 = self.upconv1(d2)
        a1 = self.att1(e1, d2)
        d1 = self.fusion1(d1, a1)
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))
        
        # 多尺度特征聚合
        msf = self.msfa([d1, d2, d3, e4])
        
        # 确保输出尺寸与输入一致
        if msf.size()[2:] != input_size:
            msf = F.interpolate(msf, size=input_size, mode='bilinear', align_corners=True)
            
        # 最终输出
        out = self.final(msf)
        
        return out 