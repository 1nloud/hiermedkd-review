"""
LiteMedFormer-S student architecture.

Compact hybrid convolution-Transformer student for efficient deployment.
Uses depthwise-separable convolution at high resolution and shifted-window attention at low resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution block.
    
    Structure: DWConv -> BN -> GELU -> PWConv -> BN -> GELU
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
        padding: int = 1
    ):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride,
            padding, groups=in_channels, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.act(x)
        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.act(x)
        return x


class WindowAttentionLite(nn.Module):
    """
    Lightweight window attention for student model.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        qkv_bias: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        # Relative position bias (simplified)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        
        return x


class TransformerBlockLite(nn.Module):
    """
    Lightweight Transformer block with window attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 2.0,
        shift_size: int = 0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttentionLite(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        assert L == H * W
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Pad if needed
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = self._window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Attention
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self._window_reverse(attn_windows, self.window_size, Hp, Wp)
        
        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x
    
    def _window_partition(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows
    
    def _window_reverse(self, windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, -1)
        return x


class LiteMedFormerSEncoder(nn.Module):
    """
    LiteMedFormer-S encoder.
    
    Configuration at 224x224:
    - Stem: 4x4 s4 Conv -> 32 channels -> 56x56
    - Stage 1: DW block x2 -> 32 channels -> 56x56
    - Stage 2: s2 DW block + DW block -> 64 channels -> 28x28
    - Stage 3: LN/s2 + window blocks x2 (2 heads) -> 96 channels -> 14x14
    - Stage 4: LN/s2 + window blocks x2 (4 heads) -> 192 channels -> 7x7
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        encoder_channels: List[int] = [32, 64, 96, 192]
    ):
        super().__init__()
        self.encoder_channels = encoder_channels
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, encoder_channels[0], kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(encoder_channels[0]),
            nn.GELU()
        )
        
        # Stage 1: DW blocks x2
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(encoder_channels[0], encoder_channels[0]),
            DepthwiseSeparableConv(encoder_channels[0], encoder_channels[0])
        )
        
        # Stage 2: s2 DW block + DW block
        self.stage2_down = DepthwiseSeparableConv(encoder_channels[0], encoder_channels[1], stride=2)
        self.stage2 = DepthwiseSeparableConv(encoder_channels[1], encoder_channels[1])
        
        # Stage 3: Transformer blocks x2
        self.stage3_down = nn.Sequential(
            nn.LayerNorm(encoder_channels[1]),
            DepthwiseSeparableConv(encoder_channels[1], encoder_channels[2], stride=2)
        )
        self.stage3 = nn.ModuleList([
            TransformerBlockLite(encoder_channels[2], num_heads=2, window_size=7, shift_size=0),
            TransformerBlockLite(encoder_channels[2], num_heads=2, window_size=7, shift_size=3)
        ])
        
        # Stage 4: Transformer blocks x2
        self.stage4_down = nn.Sequential(
            nn.LayerNorm(encoder_channels[2]),
            DepthwiseSeparableConv(encoder_channels[2], encoder_channels[3], stride=2)
        )
        self.stage4 = nn.ModuleList([
            TransformerBlockLite(encoder_channels[3], num_heads=4, window_size=7, shift_size=0),
            TransformerBlockLite(encoder_channels[3], num_heads=4, window_size=7, shift_size=3)
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, C, H, W)
        
        Returns:
            List of features at different stages
        """
        features = []
        
        # Stem
        x = self.stem(x)  # (B, 32, H/4, W/4)
        features.append(x)
        
        # Stage 1
        x = self.stage1(x)
        features.append(x)
        
        # Stage 2
        x = self.stage2_down(x)
        x = self.stage2(x)
        features.append(x)
        
        # Stage 3
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.stage3_down[0](x)
        x = x.view(B, C, H, W)
        x = self.stage3_down[1](x)
        
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for blk in self.stage3:
            x = blk(x, H, W)
        x = x.view(B, C, H, W)
        features.append(x)
        
        # Stage 4
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.stage4_down[0](x)
        x = x.view(B, C, H, W)
        x = self.stage4_down[1](x)
        
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for blk in self.stage4:
            x = blk(x, H, W)
        x = x.view(B, C, H, W)
        features.append(x)
        
        return features


class DecoderBlock(nn.Module):
    """
    Decoder block with skip connection.
    
    Bilinear upsampling -> concat with skip -> 2x Conv3x3
    """
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class LiteMedFormerS(nn.Module):
    """
    LiteMedFormer-S student model.
    
    Compact hybrid convolution-Transformer architecture:
    - Encoder: DW conv at high res, window attention at low res
    - Decoder: Skip connections with bilinear upsampling
    - Head: 1x1 conv at stride 4, restored to input grid
    
    Total parameters: ~1.36M
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 8,
        encoder_channels: List[int] = [32, 64, 96, 192],
        decoder_channels: List[int] = [192, 96, 64, 32]
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = LiteMedFormerSEncoder(in_channels, encoder_channels)
        
        # Decoder
        self.decoder = nn.ModuleList([
            DecoderBlock(encoder_channels[3], encoder_channels[2], decoder_channels[0]),
            DecoderBlock(decoder_channels[0], encoder_channels[1], decoder_channels[1]),
            DecoderBlock(decoder_channels[1], encoder_channels[0], decoder_channels[2])
        ])
        
        # Final head
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels[2], decoder_channels[3], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.GELU(),
            nn.Conv2d(decoder_channels[3], num_classes, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, C, H, W)
        
        Returns:
            Dictionary containing:
            - 'logits': output logits (B, num_classes, H, W)
            - 'features': list of encoder features
        """
        input_size = x.shape[-2:]
        
        # Encoder
        features = self.encoder(x)
        
        # Decoder
        x = features[-1]
        for i, decoder_block in enumerate(self.decoder):
            skip = features[-(i + 2)]
            x = decoder_block(x, skip)
        
        # Head
        logits = self.head(x)
        
        # Restore to input resolution
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        return {
            'logits': logits,
            'features': features
        }