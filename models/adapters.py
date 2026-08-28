"""
Stage alignment adapters for heterogeneous teacher-student feature matching.

These adapters align student features to teacher's channel dimensions for feature imitation.
Adapters are discarded after training; only the student is exported for inference.
"""

import torch
import torch.nn as nn
from typing import List


class Conv1x1Adapter(nn.Module):
    """
    1x1 convolution adapter for channel alignment.
    
    Projects student features to match teacher's channel dimension.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bn: bool = True,
        use_relu: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=not use_bn)
        
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class LinearAdapter(nn.Module):
    """
    Linear adapter for channel alignment (operates on flattened features).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, H*W, C) -> (B, H*W, C_out) -> (B, C_out, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2).view(B, -1, H, W)
        return x


class StageAdapters(nn.Module):
    """
    Collection of stage adapters for aligning student features to teacher features.
    
    Creates one adapter per stage to match channel dimensions.
    Adapters are only used during training for feature imitation.
    """
    
    def __init__(
        self,
        student_channels: List[int],
        teacher_channels: List[int],
        adapter_type: str = "conv1x1"
    ):
        """
        Args:
            student_channels: List of student channel dimensions at each stage
            teacher_channels: List of teacher channel dimensions at each stage
            adapter_type: Type of adapter ('conv1x1' or 'linear')
        """
        super().__init__()
        assert len(student_channels) == len(teacher_channels), \
            "Student and teacher must have same number of stages"
        
        self.adapters = nn.ModuleList()
        
        for s_ch, t_ch in zip(student_channels, teacher_channels):
            if s_ch == t_ch:
                # No adapter needed if channels match
                self.adapters.append(nn.Identity())
            else:
                if adapter_type == "conv1x1":
                    adapter = Conv1x1Adapter(s_ch, t_ch, use_bn=True, use_relu=False)
                elif adapter_type == "linear":
                    adapter = LinearAdapter(s_ch, t_ch)
                else:
                    raise ValueError(f"Unknown adapter type: {adapter_type}")
                self.adapters.append(adapter)
    
    def forward(self, student_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            student_features: List of student feature maps
        
        Returns:
            List of aligned feature maps matching teacher's channel dimensions
        """
        aligned_features = []
        for feat, adapter in zip(student_features, self.adapters):
            aligned_features.append(adapter(feat))
        return aligned_features