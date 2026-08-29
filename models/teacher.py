"""
Swin-style U-Net teacher architecture.

Clean-room implementation inspired by Swin-Unet for knowledge distillation.
Configuration: embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24], window_size=7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention (W-MSA).
    
    Reference: Liu et al., "Swin Transformer", ICCV 2021
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Get pair-wise relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (num_windows * B, window_size * window_size, C)
            mask: Attention mask for shifted window (optional)
        
        Returns:
            Output tensor of shape (num_windows * B, window_size * window_size, C)
        """
        B_, N, C = x.shape
        
        # QKV
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
        
        attn = self.attn_drop(attn)
        
        # Output
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block with window attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, num_heads, window_size, qkv_bias, attn_drop, drop
        )
        
        self.drop_path = nn.Identity() if drop_path <= 0 else nn.Dropout(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, H*W, C)
            H: Height
            W: Width
        
        Returns:
            Output tensor of shape (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Pad feature maps if divisible by window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            image_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = h_slices
            region_id = 0
            for h_slice in h_slices:
                for w_slice in w_slices:
                    image_mask[:, h_slice, w_slice, :] = region_id
                    region_id += 1
            mask_windows = window_partition(image_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size ** 2)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        else:
            shifted_x = x
            attn_mask = None
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition into windows.
    
    Args:
        x: (B, H, W, C)
        window_size: window size
    
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Merge windows back to feature map.
    
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: window size
        H: height
        W: width
    
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class PatchMerging(nn.Module):
    """
    Patch merging layer to downsample feature maps.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, H*W, C)
            H: height
            W: width
        
        Returns:
            x: (B, H/2*W/2, 2*C)
            H/2, W/2
        """
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        
        x = x.view(B, H, W, C)
        
        # Pad if needed
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        
        H_out = (H + 1) // 2
        W_out = (W + 1) // 2
        
        return x, H_out, W_out


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer for one stage.
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        downsample: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        # Build transformer blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path
            )
            for i in range(depth)
        ])
        
        # Patch merging (downsample)
        if downsample:
            self.downsample = PatchMerging(dim)
        else:
            self.downsample = None
    
    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, int, int]]]:
        """
        Args:
            x: (B, H*W, C)
            H: height
            W: width
        
        Returns:
            x: output features
            (x_downsampled, H_out, W_out) if downsample else None
        """
        for blk in self.blocks:
            x = blk(x, H, W)
        
        if self.downsample is not None:
            x_down, H_out, W_out = self.downsample(x, H, W)
            return x, (x_down, H_out, W_out)
        else:
            return x, None


class SwinUNETeacher(nn.Module):
    """
    Swin-style U-Net teacher for knowledge distillation.
    
    Configuration:
    - Encoder widths: (96, 192, 384, 768)
    - Depths: (2, 2, 6, 2)
    - Head counts: (3, 6, 12, 24)
    - Window size: 7x7
    - MLP ratio: 4
    - Encoder strides: (4, 8, 16, 32)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 8,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=4, stride=4
        )
        self.patch_norm = nn.LayerNorm(embed_dim)
        
        # Encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.encoder_layers = nn.ModuleList()
        self.encoder_strides = [4, 8, 16, 32]
        self.encoder_dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        
        for i in range(4):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=(i < 3)
            )
            self.encoder_layers.append(layer)
        
        # Decoder (convolutional U-Net style)
        self.decoder_upsample = nn.ModuleList([
            nn.ConvTranspose2d(
                self.encoder_dims[3 - i], self.encoder_dims[2 - i],
                kernel_size=2, stride=2
            )
            for i in range(3)
        ])
        
        self.decoder_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * self.encoder_dims[2 - i], self.encoder_dims[2 - i], 3, padding=1),
                nn.BatchNorm2d(self.encoder_dims[2 - i]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.encoder_dims[2 - i], self.encoder_dims[2 - i], 3, padding=1),
                nn.BatchNorm2d(self.encoder_dims[2 - i]),
                nn.ReLU(inplace=True)
            )
            for i in range(3)
        ])
        
        # Final head
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.encoder_dims[0], self.encoder_dims[0] // 2, 3, padding=1),
            nn.BatchNorm2d(self.encoder_dims[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.encoder_dims[0] // 2, num_classes, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Dictionary containing:
            - 'logits': output logits (B, num_classes, H, W)
            - 'features': list of encoder features at different stages
        """
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/4, W/4)
        x = x.flatten(2).transpose(1, 2)  # (B, H/4*W/4, embed_dim)
        x = self.patch_norm(x)
        
        H0, W0 = H // 4, W // 4
        
        # Encoder with skip connections
        encoder_features = []
        
        for i, layer in enumerate(self.encoder_layers):
            x, down_out = layer(x, H0, W0)
            encoder_features.append(x.view(B, H0, W0, -1).permute(0, 3, 1, 2))
            
            if down_out is not None:
                x, H0, W0 = down_out
        
        # Decoder
        x = x.view(B, H0, W0, -1).permute(0, 3, 1, 2)
        
        for i in range(3):
            x = self.decoder_upsample[i](x)
            skip = encoder_features[2 - i]
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_conv[i](x)
        
        # Final upsampling to input resolution
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        logits = self.final_conv(x)
        
        return {
            'logits': logits,
            'features': encoder_features
        }
