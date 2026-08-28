"""
HierMedKD: Hierarchical Boundary-Aware Distillation Losses

This module implements the three core distillation losses:
1. Normalized feature imitation loss
2. Activation energy transfer loss
3. Boundary-weighted KL divergence loss with entropy gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import numpy as np


class NormalizedFeatureLoss(nn.Module):
    """
    Normalized hierarchical feature imitation loss.
    
    For a feature vector at location i, define v(f_i) = f_i / max(||f_i||_2, epsilon).
    The loss averages over batch items, spatial locations, and stages.
    
    Reference: Eq. (1) in the paper
    """
    
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(
        self,
        teacher_features: List[torch.Tensor],
        student_features: List[torch.Tensor],
        stage_weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        Args:
            teacher_features: List of teacher feature maps at different stages
            student_features: List of student feature maps at different stages
            stage_weights: Optional weights for each stage (default: equal)
        
        Returns:
            Scalar loss value
        """
        assert len(teacher_features) == len(student_features), \
            "Teacher and student features must have same number of stages"
        
        num_stages = len(teacher_features)
        if stage_weights is None:
            stage_weights = [1.0 / num_stages] * num_stages
        else:
            assert len(stage_weights) == num_stages
            total_weight = sum(stage_weights)
            stage_weights = [w / total_weight for w in stage_weights]
        
        total_loss = 0.0
        
        for t_feat, s_feat, weight in zip(teacher_features, student_features, stage_weights):
            # Normalize features: v(f) = f / max(||f||_2, epsilon)
            # Compute L2 norm per spatial location
            t_norm = torch.norm(t_feat, p=2, dim=1, keepdim=True)
            s_norm = torch.norm(s_feat, p=2, dim=1, keepdim=True)
            
            t_normalized = t_feat / torch.clamp(t_norm, min=self.epsilon)
            s_normalized = s_feat / torch.clamp(s_norm, min=self.epsilon)
            
            # Compute MSE loss and average over spatial locations
            feat_loss = F.mse_loss(t_normalized, s_normalized, reduction='mean')
            
            total_loss += weight * feat_loss
        
        return total_loss


class ActivationEnergyLoss(nn.Module):
    """
    Architecture-agnostic activation energy transfer loss.
    
    Uses activation energy instead of raw attention matrices for architecture-agnostic transfer.
    For F in R^{C x H x W}, let A(F)_{ij} = sum_c |F_{cij}|^2.
    The normalized energy is A_bar = A / max(||A||_2, epsilon).
    
    Reference: Eq. (2) in the paper
    """
    
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def compute_activation_energy(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Compute activation energy map.
        
        Args:
            feat: Feature map of shape (B, C, H, W)
        
        Returns:
            Energy map of shape (B, H, W)
        """
        # A(F)_{ij} = sum_c |F_{cij}|^2
        energy = torch.sum(feat ** 2, dim=1)  # (B, H, W)
        return energy
    
    def forward(
        self,
        teacher_features: List[torch.Tensor],
        student_features: List[torch.Tensor],
        stage_weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        Args:
            teacher_features: List of teacher feature maps at different stages
            student_features: List of student feature maps at different stages
            stage_weights: Optional weights for each stage (default: equal)
        
        Returns:
            Scalar loss value
        """
        assert len(teacher_features) == len(student_features), \
            "Teacher and student features must have same number of stages"
        
        num_stages = len(teacher_features)
        if stage_weights is None:
            stage_weights = [1.0 / num_stages] * num_stages
        else:
            assert len(stage_weights) == num_stages
            total_weight = sum(stage_weights)
            stage_weights = [w / total_weight for w in stage_weights]
        
        total_loss = 0.0
        
        for t_feat, s_feat, weight in zip(teacher_features, student_features, stage_weights):
            # Compute activation energy
            t_energy = self.compute_activation_energy(t_feat)  # (B, H_t, W_t)
            s_energy = self.compute_activation_energy(s_feat)  # (B, H_s, W_s)
            
            # Resize student energy to teacher's spatial grid
            s_energy_resized = F.interpolate(
                s_energy.unsqueeze(1),
                size=t_energy.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # (B, H_t, W_t)
            
            # Normalize energy maps
            t_norm = torch.norm(t_energy, p=2, dim=(1, 2), keepdim=True)
            s_norm = torch.norm(s_energy_resized, p=2, dim=(1, 2), keepdim=True)
            
            t_normalized = t_energy / torch.clamp(t_norm, min=self.epsilon)
            s_normalized = s_energy_resized / torch.clamp(s_norm, min=self.epsilon)
            
            # Compute MSE loss (no spatial averaging needed due to unit norm)
            att_loss = F.mse_loss(t_normalized, s_normalized, reduction='mean')
            
            total_loss += weight * att_loss
        
        return total_loss


class BoundaryWeightedKLLoss(nn.Module):
    """
    Multiclass boundary-weighted soft-label loss with entropy gating.
    
    The boundary location comes from ground truth y, the reliability gate
    and soft distribution come from teacher T.
    
    Reference: Eq. (3)-(5) in the paper
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        beta: float = 4.0,
        kernel_size: int = 3,
        epsilon: float = 1e-6,
        confidence_gating: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.beta = beta
        self.kernel_size = kernel_size
        self.epsilon = epsilon
        self.confidence_gating = confidence_gating
        
        # Morphological kernel for boundary extraction
        self.register_buffer(
            'kernel',
            torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        )
    
    def extract_boundary_mask(
        self,
        label: torch.Tensor,
        ignore_label: int = 255
    ) -> torch.Tensor:
        """
        Extract multiclass boundary mask using morphological operations.
        
        Args:
            label: Label tensor of shape (B, H, W) with values in [0, C-1]
            ignore_label: Label value to ignore
        
        Returns:
            Boundary mask of shape (B, H, W) with 1 at boundary, 0 elsewhere
        """
        B, H, W = label.shape
        num_classes = label.max().item() + 1
        
        # Create one-hot masks for each foreground class
        boundary_masks = []
        for c in range(1, num_classes):  # Skip background (class 0)
            class_mask = (label == c).float().unsqueeze(1)  # (B, 1, H, W)
            
            # Morphological dilation
            dilated = F.conv2d(
                class_mask,
                self.kernel,
                padding=(self.kernel_size // 2, self.kernel_size // 2)
            )
            
            # Morphological erosion
            eroded = 1 - F.conv2d(
                1 - class_mask,
                self.kernel,
                padding=(self.kernel_size // 2, self.kernel_size // 2)
            )
            
            # Boundary = dilated - eroded
            class_boundary = (dilated - eroded > 0).float()
            boundary_masks.append(class_boundary)
        
        # Union of all class boundaries
        boundary_union = torch.stack(boundary_masks, dim=1).max(dim=1)[0]  # (B, H, W)
        
        return boundary_union
    
    def compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of probability distribution.
        
        Args:
            probs: Probability tensor of shape (B, C, H, W)
        
        Returns:
            Entropy of shape (B, H, W)
        """
        # H(p) = -sum_c p_c * log(p_c + epsilon)
        entropy = -torch.sum(probs * torch.log(probs + self.epsilon), dim=1)
        return entropy
    
    def forward(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_label: int = 255
    ) -> torch.Tensor:
        """
        Args:
            teacher_logits: Teacher logits of shape (B, C, H, W)
            student_logits: Student logits of shape (B, C, H, W)
            labels: Ground truth labels of shape (B, H, W)
            ignore_label: Label value to ignore
        
        Returns:
            Scalar loss value
        """
        B, C, H, W = teacher_logits.shape
        
        # Compute soft distributions with temperature
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_probs = F.softmax(student_logits / self.temperature, dim=1)
        
        # Extract boundary mask from ground truth
        boundary_mask = self.extract_boundary_mask(labels, ignore_label)
        
        # Valid mask (not ignore label)
        valid_mask = (labels != ignore_label).float()
        
        # Compute teacher entropy and confidence gate
        if self.confidence_gating:
            teacher_entropy = self.compute_entropy(teacher_probs)
            # r = clip(1 - H / log(C), [0, 1])
            confidence_gate = torch.clamp(
                1 - teacher_entropy / np.log(C),
                min=0.0,
                max=1.0
            )
        else:
            confidence_gate = torch.ones_like(boundary_mask)
        
        # Compute weights: w = v * [1 + beta * B * r]
        weights = valid_mask * (1 + self.beta * boundary_mask * confidence_gate)
        
        # Compute KL divergence per pixel
        kl_div = teacher_probs * torch.log(
            (teacher_probs + self.epsilon) / (student_probs + self.epsilon)
        )
        kl_div = torch.sum(kl_div, dim=1)  # (B, H, W)
        
        # Weighted KL loss with normalization
        weighted_kl = weights * kl_div
        total_weight = torch.sum(weights) + self.epsilon
        
        loss = torch.sum(weighted_kl) / total_weight
        
        return loss


class HierMedKDLoss(nn.Module):
    """
    Complete HierMedKD loss combining all three distillation terms.
    
    L = L_seg + lambda_f * L_feat + lambda_a * L_att + lambda_k * L_bkl
    """
    
    def __init__(
        self,
        lambda_feat: float = 1.0,
        lambda_att: float = 1.0,
        lambda_kl: float = 1.0,
        temperature: float = 4.0,
        beta: float = 4.0,
        kernel_size: int = 3,
        epsilon: float = 1e-6,
        confidence_gating: bool = True
    ):
        super().__init__()
        self.lambda_feat = lambda_feat
        self.lambda_att = lambda_att
        self.lambda_kl = lambda_kl
        
        self.feature_loss = NormalizedFeatureLoss(epsilon)
        self.activation_loss = ActivationEnergyLoss(epsilon)
        self.boundary_loss = BoundaryWeightedKLLoss(
            temperature, beta, kernel_size, epsilon, confidence_gating
        )
    
    def forward(
        self,
        teacher_features: List[torch.Tensor],
        student_features: List[torch.Tensor],
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_label: int = 255,
        stage_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            teacher_features: List of teacher feature maps
            student_features: List of student feature maps
            teacher_logits: Teacher output logits
            student_logits: Student output logits
            labels: Ground truth labels
            ignore_label: Label value to ignore
            stage_weights: Optional weights for each stage
        
        Returns:
            Dictionary of loss components
        """
        # Compute individual losses
        feat_loss = self.feature_loss(teacher_features, student_features, stage_weights)
        att_loss = self.activation_loss(teacher_features, student_features, stage_weights)
        bkl_loss = self.boundary_loss(teacher_logits, student_logits, labels, ignore_label)
        
        # Total distillation loss
        kd_loss = (
            self.lambda_feat * feat_loss +
            self.lambda_att * att_loss +
            self.lambda_kl * bkl_loss
        )
        
        return {
            'feat_loss': feat_loss,
            'att_loss': att_loss,
            'bkl_loss': bkl_loss,
            'kd_loss': kd_loss
        }