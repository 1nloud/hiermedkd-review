"""
Supervised loss functions for medical image segmentation.

Implements cross-entropy and Dice loss for supervised training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice loss for foreground macro averaging.
    
    Reference: Eq. (6) in the paper
    """
    
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_label: int = 255
    ) -> torch.Tensor:
        """
        Args:
            logits: Model logits of shape (B, C, H, W)
            labels: Ground truth labels of shape (B, H, W)
            ignore_label: Label value to ignore
        
        Returns:
            Scalar Dice loss (foreground macro average)
        """
        B, C, H, W = logits.shape
        
        # Use soft probabilities so the Dice term remains differentiable.
        probs = F.softmax(logits, dim=1)
        
        # Valid mask
        valid_mask = (labels != ignore_label)
        
        # Compute Dice for each foreground class
        dice_scores = []
        for c in range(1, C):  # Skip background
            # Binary masks for class c
            pred_mask = probs[:, c]
            label_mask = (labels == c).float()
            
            # Apply valid mask
            pred_mask = pred_mask * valid_mask.float()
            label_mask = label_mask * valid_mask.float()
            
            # Dice coefficient
            intersection = (pred_mask * label_mask).sum()
            union = pred_mask.sum() + label_mask.sum()
            
            dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
            dice_scores.append(dice)
        
        # Macro average over foreground classes
        dice_loss = 1 - torch.stack(dice_scores).mean()
        
        return dice_loss


class SupervisedLoss(nn.Module):
    """
    Combined supervised loss: cross-entropy + Dice.
    
    Reference: Eq. (7)-(8) in the paper
    """
    
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.dice_loss = DiceLoss(epsilon)
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_label: int = 255
    ) -> dict:
        """
        Args:
            logits: Model logits of shape (B, C, H, W)
            labels: Ground truth labels of shape (B, H, W)
            ignore_label: Label value to ignore
        
        Returns:
            Dictionary of loss components
        """
        ce_loss = self.ce_loss(logits, labels)
        dice_loss = self.dice_loss(logits, labels, ignore_label)
        
        total_loss = ce_loss + dice_loss
        
        return {
            'ce_loss': ce_loss,
            'dice_loss': dice_loss,
            'seg_loss': total_loss
        }
