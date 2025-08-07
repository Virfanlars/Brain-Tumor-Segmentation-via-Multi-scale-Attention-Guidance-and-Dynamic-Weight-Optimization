import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange
from .attention import CSDA, LinearAttention

class OverlapPatchEmbed(nn.Module):
    """ 重叠补丁嵌入
    将图像分成重叠的补丁，通过卷积进行嵌入
    """
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=patch_size//2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, H=None, W=None):
        """
        输入:
            x: (B, C, H, W) 或 (B, N, C)
            H, W: 可选，当x为序列形式时需要提供
        """
        # 如果x是序列形式，将其重塑为空间形式
        if len(x.shape) == 3:
            B, N, C = x.shape
            if H is None or W is None:
                # 假设序列是正方形，计算H和W
                H = W = int(N ** 0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # 进行卷积投影
        x = self.proj(x)  # [B, C, H, W]
        _, _, H, W = x.shape
        
        # 转回序列形式
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x = self.norm(x)
        
        return x, H, W

class MixFFN(nn.Module):
    """ MixFFN: 混合前馈网络
    结合了MLP和卷积操作
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, 
                               padding=1, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EfficientAttention(nn.Module):
    """ 高效注意力机制
    使用线性注意力降低计算复杂度
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"维度 {dim} 必须被头数 {num_heads} 整除"
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.sr = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        shortcut = x
        
        # 空间还原模块
        x_ = x.transpose(1, 2).view(B, C, H, W)
        x_ = self.sr(x_)
        x_ = x_.reshape(B, C, -1).transpose(1, 2)
        x = x + self.act(self.norm(x_))
        
        # 高效自注意力
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 线性注意力
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        
        context = k.transpose(-2, -1) @ v
        attn = (q @ context).transpose(1, 2).reshape(B, N, C)
        attn = self.attn_drop(attn)
        
        x = self.proj(attn)
        x = self.proj_drop(x)
        
        # 残差连接
        x = x + shortcut
        
        return x

class TransformerBlock(nn.Module):
    """ Transformer块
    结合了注意力机制和MixFFN
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EfficientAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        # 添加dropout path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixFFN(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
        # 添加CSDA模块
        self.csda = CSDA(dim, reduction_ratio=16)

    def forward(self, x, H, W):
        # 检查输入维度与norm层期望的维度是否匹配
        _, _, C = x.shape
        if C != self.norm1.normalized_shape[0]:
            # 如果不匹配，调整norm层的维度
            device = x.device
            self.norm1 = nn.LayerNorm(C).to(device)
            self.norm2 = nn.LayerNorm(C).to(device)
            self.attn = EfficientAttention(
                C, num_heads=self.attn.num_heads, qkv_bias=self.attn.qkv.bias is not None,
                attn_drop=self.attn.attn_drop.p, proj_drop=self.attn.proj_drop.p).to(device)
            self.mlp = MixFFN(
                in_features=C, 
                hidden_features=int(C * 4), 
                drop=self.mlp.drop.p).to(device)
            self.csda = CSDA(C, reduction_ratio=16).to(device)
        
        # 自注意力
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        
        # MixFFN
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        
        # 转回空间域，应用CSDA
        x_spatial = x.transpose(1, 2).reshape(-1, x.size(2), H, W)
        x_spatial = self.csda(x_spatial)
        
        # 转回序列形式
        x = x_spatial.flatten(2).transpose(1, 2)
        
        return x

class MixViTStage(nn.Module):
    """ MixViT的单个阶段
    """
    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., 
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, 
                 downsample=None, use_csda=True):
        super().__init__()
        self.use_csda = use_csda
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                drop=drop, attn_drop=attn_drop, 
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.downsample = downsample
        self.dim = dim

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        
        return x, H, W

class MixViTEncoder(nn.Module):
    """ MixViT编码器
    多尺度特征金字塔编码器
    """
    def __init__(self, in_chans=1, embed_dims=[64, 128, 256, 512], 
                 depths=[3, 4, 6, 3], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], 
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, 
                 norm_layer=nn.LayerNorm, patch_sizes=[7, 3, 3, 3], 
                 strides=[4, 2, 2, 2], sr_ratios=[8, 4, 2, 1], use_csda=True):
        super().__init__()
        self.depths = depths
        self.num_stages = len(depths)
        self.embed_dims = embed_dims
        
        # 随深度线性增加dropout rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        # 构建每个阶段
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            patch_embed = OverlapPatchEmbed(
                patch_size=patch_sizes[i], stride=strides[i], 
                in_chans=in_chans if i == 0 else embed_dims[i-1], 
                embed_dim=embed_dims[i])
            
            stage = MixViTStage(
                dim=embed_dims[i], depth=depths[i], num_heads=num_heads[i], 
                mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, drop=drop_rate, 
                attn_drop=attn_drop_rate, drop_path=dpr[cur:cur+depths[i]], 
                norm_layer=norm_layer, downsample=patch_embed if i < self.num_stages-1 else None,
                use_csda=use_csda)
            
            self.stages.append(stage)
            cur += depths[i]
            
            # 第一个阶段后切换输入通道数
            in_chans = embed_dims[i]
        
        # 应用层归一化到最后的输出
        self.norm = norm_layer(embed_dims[-1])

    def forward(self, x):
        # 记录每个阶段的特征，便于后续采用跳跃连接
        features = []
        
        # 第一个阶段：直接从输入图像开始
        # 第一个Patch Embedding
        patch_embed = self.stages[0].downsample
        x, H, W = patch_embed(x)
        
        # 通过第一阶段的Transformer blocks
        for blk in self.stages[0].blocks:
            x = blk(x, H, W)
        
        # 将特征转回空间域并保存
        feat = x.transpose(1, 2).reshape(-1, self.embed_dims[0], H, W)
        features.append(feat)
        
        # 处理其余阶段
        for i in range(1, len(self.stages)):
            # 获取当前阶段的维度
            current_dim = self.embed_dims[i]
            prev_dim = self.embed_dims[i-1]
            
            # 从空间特征转换为当前阶段的特征尺寸
            # 使用1x1卷积调整通道数
            adapter = nn.Conv2d(prev_dim, current_dim, kernel_size=1).to(feat.device)
            feat = adapter(feat)
            
            # 下采样（如果需要）
            if i < len(self.stages) - 1:
                feat = F.avg_pool2d(feat, kernel_size=2)
                
            # 获取新的H和W
            _, _, H, W = feat.shape
                
            # 转回序列形式
            x = feat.flatten(2).transpose(1, 2)
            
            # 通过当前阶段的Transformer blocks
            for blk in self.stages[i].blocks:
                x = blk(x, H, W)
            
            # 将特征转回空间域并保存
            feat = x.transpose(1, 2).reshape(-1, current_dim, H, W)
            features.append(feat)
        
        return features 