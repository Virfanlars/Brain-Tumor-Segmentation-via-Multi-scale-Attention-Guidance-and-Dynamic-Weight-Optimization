import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PatchEmbed(nn.Module):
    """将图像分割成patch并进行线性嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"输入图像大小({H}*{W})与模型预期({self.img_size[0]}*{self.img_size[1]})不一致"
        
        # (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(x)
        # (B, embed_dim, H', W') -> (B, embed_dim, H'*W')
        x = x.flatten(2)
        # (B, embed_dim, H'*W') -> (B, H'*W', embed_dim)
        x = x.transpose(1, 2)
        
        return x

class Attention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, dim, n_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        """x: (B, N, C)"""
        B, N, C = x.shape
        
        # (B, N, 3*C) -> (B, N, 3, n_heads, head_dim) -> (3, B, n_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, n_heads, N, head_dim)
        
        # (B, n_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # (B, n_heads, N, head_dim) -> (B, N, n_heads, head_dim) -> (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(TransformerBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        """x: (B, N, C)"""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DecoderBlock(nn.Module):
    """解码器块"""
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class TransUNet(nn.Module):
    """基于Transformer的U型网络"""
    def __init__(self, in_channels=1, out_channels=1, img_size=224, patch_size=16, 
                 embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., features=[64, 128, 256, 512]):
        super(TransUNet, self).__init__()
        
        # CNN编码器部分
        self.encoder_cnn = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        
        # 初始卷积
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        # CNN编码器
        in_channels = features[0]
        for feature in features[1:]:
            self.encoder_pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.encoder_cnn.append(nn.Sequential(
                nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True)
            ))
            in_channels = feature
        
        # Transformer编码器
        self.patch_embed = PatchEmbed(
            img_size=img_size // 16,  # 因为经过了4次下采样
            patch_size=patch_size // 16,
            in_channels=features[-1],
            embed_dim=embed_dim
        )
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate
            ) for _ in range(depth)
        ])
        
        # 将Transformer特征转换回CNN特征
        self.conv_transpose = nn.ConvTranspose2d(
            embed_dim, features[-1], kernel_size=patch_size // 16, stride=patch_size // 16
        )
        
        # 解码器
        self.decoder = nn.ModuleList()
        features = features[::-1]  # 反转特征列表
        
        for i in range(len(features) - 1):
            self.decoder.append(DecoderBlock(features[i], features[i + 1]))
        
        # 最终输出层
        self.final_conv = nn.Conv2d(features[-1], out_channels, kernel_size=1)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        # 初始化Transformer部分的权重
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        # 记录原始大小
        B, C, H, W = x.shape
        
        # CNN编码部分
        skip_connections = []
        
        # 初始卷积
        x = self.initial_conv(x)
        skip_connections.append(x)
        
        # CNN编码层
        for i, (pool, conv) in enumerate(zip(self.encoder_pools, self.encoder_cnn)):
            x = pool(x)
            x = conv(x)
            if i < len(self.encoder_pools) - 1:
                skip_connections.append(x)
        
        # 获取当前特征图大小
        _, _, H_enc, W_enc = x.shape
        
        # 动态调整Transformer的输入大小
        self.patch_embed.img_size = (H_enc, W_enc)
        
        # Transformer编码部分
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # 保存原始大小用于后续恢复
        x_reshape = x
        
        # 动态调整位置编码大小以匹配patch数量
        n_patches = x.shape[1]
        if n_patches != self.pos_embed.shape[1]:
            # 调整位置编码大小
            pos_embed = F.interpolate(
                self.pos_embed.reshape(1, int(self.patch_embed.n_patches**0.5), int(self.patch_embed.n_patches**0.5), -1).permute(0, 3, 1, 2),
                size=(int(n_patches**0.5), int(n_patches**0.5)),
                mode='bilinear',
                align_corners=True
            ).permute(0, 2, 3, 1).reshape(1, n_patches, -1)
        else:
            pos_embed = self.pos_embed
        
        # 转换为patch并加入位置编码
        x = x + pos_embed
        x = self.pos_drop(x)
        
        # 通过Transformer块
        for block in self.transformer_blocks:
            x = block(x)
        
        # 恢复为图像格式
        x = x.transpose(1, 2).reshape(B, -1, int(np.sqrt(x.size(1))), int(np.sqrt(x.size(1))))
        x = self.conv_transpose(x)
        
        # 确保大小一致
        if x.shape != x_reshape.shape:
            x = F.interpolate(x, size=(H_enc, W_enc), mode="bilinear", align_corners=True)
        
        # 解码器部分
        skip_connections = skip_connections[::-1]  # 反转以便从最深层开始
        
        for i, decoder in enumerate(self.decoder):
            skip = skip_connections[i] if i < len(skip_connections) else None
            x = decoder(x, skip)
        
        # 最终输出层
        x = self.final_conv(x)
        
        # 调整到原始图像大小
        if x.shape[2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)
        
        return x 