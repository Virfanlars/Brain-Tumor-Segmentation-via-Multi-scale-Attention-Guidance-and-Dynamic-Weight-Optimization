import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器部分
        for feature in features:
            self.encoder.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # 解码器部分
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(ConvBlock(feature*2, feature))

        # 中间的瓶颈层
        self.bottleneck = ConvBlock(features[-1], features[-1]*2)
        
        # 最终的1x1卷积层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # 编码器前向传播
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 解码器前向传播
        skip_connections = skip_connections[::-1]  # 反转以便从最深层开始
        
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # 上采样
            skip_connection = skip_connections[idx//2]
            
            # 处理大小不一致的情况
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
                
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip)  # 卷积块
            
        # 最终输出
        return self.final_conv(x) 