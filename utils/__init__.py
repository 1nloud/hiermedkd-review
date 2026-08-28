"""
Utility functions for metrics, transforms, and checkpoint management.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import json
import shutil


def compute_dice(
    pred: torch.Tensor,
    label: torch.Tensor,
    ignore_label: int = 255
) -> float:
    """
    Compute Dice coefficient for a single class.
    
    Args:
        pred: Prediction tensor (H, W)
        label: Ground truth tensor (H, W)
        ignore_label: Label value to ignore
    
    Returns:
        Dice coefficient
    """
    pred = pred.float()
    label = label.float()
    
    valid_mask = (label != ignore_label)
    pred = pred * valid_mask
    label = label * valid_mask
    
    intersection = (pred * label).sum()
    union = pred.sum() + label.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    return dice.item()


def compute_jaccard(
    pred: torch.Tensor,
    label: torch.Tensor,
    ignore_label: int = 255
) -> float:
    """Compute Jaccard/IoU for a single binary class mask."""
    pred = pred.bool()
    label = label.bool()
    valid_mask = label != ignore_label
    pred = pred & valid_mask
    label = label & valid_mask
    intersection = (pred & label).sum().item()
    union = (pred | label).sum().item()
    return 1.0 if union == 0 else float(intersection / union)


def compute_hd95(
    pred: np.ndarray,
    label: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0),
    ignore_label: int = 255
) -> float:
    """
    Compute 95th percentile Hausdorff distance.
    
    Args:
        pred: Prediction array (H, W) or (D, H, W)
        label: Ground truth array (H, W) or (D, H, W)
        spacing: Pixel/voxel spacing
        ignore_label: Label value to ignore
    
    Returns:
        HD95 distance
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt
    
    # Binary masks
    pred_binary = (pred > 0).astype(np.uint8)
    label_binary = (label > 0).astype(np.uint8)
    
    if np.sum(pred_binary) == 0 and np.sum(label_binary) == 0:
        return 0.0
    
    if np.sum(pred_binary) == 0 or np.sum(label_binary) == 0:
        return float('inf')
    
    pred_surface = pred_binary.astype(bool) ^ binary_erosion(pred_binary.astype(bool))
    label_surface = label_binary.astype(bool) ^ binary_erosion(label_binary.astype(bool))
    if not pred_surface.any() or not label_surface.any():
        return float('inf')

    distance_to_label = distance_transform_edt(~label_surface, sampling=spacing)
    distance_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    surface_distances = np.concatenate([
        distance_to_label[pred_surface],
        distance_to_pred[label_surface],
    ])
    hd95 = np.percentile(surface_distances, 95)
    return float(hd95)


def compute_nsd(
    pred: np.ndarray,
    label: np.ndarray,
    tolerance: float = 2.0,
    spacing: Tuple[float, ...] = (1.0, 1.0),
    ignore_label: int = 255
) -> float:
    """
    Compute Normalized Surface Dice (NSD).
    
    Args:
        pred: Prediction array
        label: Ground truth array
        tolerance: Tolerance distance in physical units
        spacing: Pixel/voxel spacing
        ignore_label: Label value to ignore
    
    Returns:
        NSD value
    """
    from scipy.ndimage import distance_transform_edt
    
    pred_binary = (pred > 0).astype(np.uint8)
    label_binary = (label > 0).astype(np.uint8)
    
    if np.sum(pred_binary) == 0 and np.sum(label_binary) == 0:
        return 1.0
    
    if np.sum(pred_binary) == 0 or np.sum(label_binary) == 0:
        return 0.0
    
    # Distance transforms
    pred_dtf = distance_transform_edt(1 - pred_binary, sampling=spacing)
    label_dtf = distance_transform_edt(1 - label_binary, sampling=spacing)
    
    # Surface points
    pred_surface = (pred_dtf <= 1.0) & (pred_binary > 0)
    label_surface = (label_dtf <= 1.0) & (label_binary > 0)
    
    # Count surface points within tolerance
    pred_to_label = np.sum(pred_surface & (label_dtf <= tolerance))
    label_to_pred = np.sum(label_surface & (pred_dtf <= tolerance))
    
    total_pred_surface = np.sum(pred_surface)
    total_label_surface = np.sum(label_surface)
    
    if total_pred_surface == 0 or total_label_surface == 0:
        return 0.0
    
    nsd = (pred_to_label + label_to_pred) / (total_pred_surface + total_label_surface)
    return float(nsd)


def compute_boundary_iou(
    pred: np.ndarray,
    label: np.ndarray,
    dilation_ratio: float = 0.02,
    ignore_label: int = 255
) -> float:
    """
    Compute Boundary IoU with specified dilation width.
    
    Args:
        pred: Prediction array (H, W)
        label: Ground truth array (H, W)
        dilation_ratio: Dilation width as ratio of image diagonal
        ignore_label: Label value to ignore
    
    Returns:
        Boundary IoU
    """
    from scipy.ndimage import binary_dilation
    
    pred_binary = (pred > 0).astype(np.uint8)
    label_binary = (label > 0).astype(np.uint8)
    
    # Compute dilation width
    h, w = pred.shape
    diagonal = np.sqrt(h ** 2 + w ** 2)
    dilation_width = int(np.round(dilation_ratio * diagonal))
    
    if dilation_width < 1:
        dilation_width = 1
    
    # Extract boundaries
    pred_dilated = binary_dilation(pred_binary, iterations=dilation_width)
    label_dilated = binary_dilation(label_binary, iterations=dilation_width)
    
    pred_boundary = pred_dilated & (~pred_binary)
    label_boundary = label_dilated & (~label_binary)
    
    # Compute IoU
    intersection = np.sum(pred_boundary & label_boundary)
    union = np.sum(pred_boundary | label_boundary)
    
    if union == 0:
        return 1.0
    
    biou = intersection / union
    return float(biou)


class SegmentationMetrics:
    """
    Compute segmentation metrics with confidence intervals.
    """
    
    def __init__(
        self,
        num_classes: int,
        ignore_label: int = 255,
        spacing: Tuple[float, ...] = (1.0, 1.0),
        nsd_tolerance: float = 2.0,
        boundary_dilation_ratio: float = 0.02
    ):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.spacing = spacing
        self.nsd_tolerance = nsd_tolerance
        self.boundary_dilation_ratio = boundary_dilation_ratio
        
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.dice_scores = {c: [] for c in range(1, self.num_classes)}
        self.jaccard_scores = {c: [] for c in range(1, self.num_classes)}
        self.hd95_scores = {c: [] for c in range(1, self.num_classes)}
        self.nsd_scores = {c: [] for c in range(1, self.num_classes)}
        self.boundary_iou_scores = {c: [] for c in range(1, self.num_classes)}
    
    def update(
        self,
        pred: torch.Tensor,
        label: torch.Tensor
    ):
        """
        Update metrics with a batch of predictions.
        
        Args:
            pred: Predictions (B, H, W) or (B, C, H, W)
            label: Ground truth (B, H, W)
        """
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        
        B = pred.shape[0]
        
        for b in range(B):
            pred_b = pred[b].cpu().numpy()
            label_b = label[b].cpu().numpy()
            
            for c in range(1, self.num_classes):
                pred_mask = (pred_b == c).astype(np.uint8)
                label_mask = (label_b == c).astype(np.uint8)
                
                # Skip if class not present
                if np.sum(label_mask) == 0:
                    continue
                
                # Dice
                dice = compute_dice(
                    torch.from_numpy(pred_mask),
                    torch.from_numpy(label_mask),
                    self.ignore_label
                )
                self.dice_scores[c].append(dice)

                jaccard = compute_jaccard(
                    torch.from_numpy(pred_mask),
                    torch.from_numpy(label_mask),
                    self.ignore_label
                )
                self.jaccard_scores[c].append(jaccard)
                
                # HD95
                try:
                    hd95 = compute_hd95(pred_mask, label_mask, self.spacing, self.ignore_label)
                    if hd95 != float('inf'):
                        self.hd95_scores[c].append(hd95)
                except:
                    pass
                
                # NSD
                try:
                    nsd = compute_nsd(pred_mask, label_mask, self.nsd_tolerance, self.spacing, self.ignore_label)
                    self.nsd_scores[c].append(nsd)
                except:
                    pass
                
                # Boundary IoU (for 2D only)
                if len(self.spacing) == 2:
                    try:
                        biou = compute_boundary_iou(pred_mask, label_mask, self.boundary_dilation_ratio, self.ignore_label)
                        self.boundary_iou_scores[c].append(biou)
                    except:
                        pass
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get computed metrics with mean and confidence intervals.
        
        Returns:
            Dictionary of metrics per class and macro average
        """
        results = {}
        
        # Compute per-class metrics
        for c in range(1, self.num_classes):
            results[f'class_{c}'] = {}
            
            if len(self.dice_scores[c]) > 0:
                dice_mean = np.mean(self.dice_scores[c])
                dice_std = np.std(self.dice_scores[c])
                results[f'class_{c}']['dice'] = dice_mean
                results[f'class_{c}']['dice_std'] = dice_std

            if len(self.jaccard_scores[c]) > 0:
                results[f'class_{c}']['jaccard'] = np.mean(self.jaccard_scores[c])
                results[f'class_{c}']['jaccard_std'] = np.std(self.jaccard_scores[c])
            
            if len(self.hd95_scores[c]) > 0:
                hd95_mean = np.mean(self.hd95_scores[c])
                hd95_std = np.std(self.hd95_scores[c])
                results[f'class_{c}']['hd95'] = hd95_mean
                results[f'class_{c}']['hd95_std'] = hd95_std
            
            if len(self.nsd_scores[c]) > 0:
                nsd_mean = np.mean(self.nsd_scores[c])
                nsd_std = np.std(self.nsd_scores[c])
                results[f'class_{c}']['nsd'] = nsd_mean
                results[f'class_{c}']['nsd_std'] = nsd_std
            
            if len(self.boundary_iou_scores[c]) > 0:
                biou_mean = np.mean(self.boundary_iou_scores[c])
                biou_std = np.std(self.boundary_iou_scores[c])
                results[f'class_{c}']['boundary_iou'] = biou_mean
                results[f'class_{c}']['boundary_iou_std'] = biou_std
        
        # Compute macro averages
        macro_dice = []
        macro_jaccard = []
        macro_hd95 = []
        macro_nsd = []
        macro_biou = []
        
        for c in range(1, self.num_classes):
            if len(self.dice_scores[c]) > 0:
                macro_dice.extend(self.dice_scores[c])
            if len(self.jaccard_scores[c]) > 0:
                macro_jaccard.extend(self.jaccard_scores[c])
            if len(self.hd95_scores[c]) > 0:
                macro_hd95.extend(self.hd95_scores[c])
            if len(self.nsd_scores[c]) > 0:
                macro_nsd.extend(self.nsd_scores[c])
            if len(self.boundary_iou_scores[c]) > 0:
                macro_biou.extend(self.boundary_iou_scores[c])
        
        results['macro'] = {}
        if len(macro_dice) > 0:
            results['macro']['dice'] = np.mean(macro_dice)
            results['macro']['dice_std'] = np.std(macro_dice)
        if len(macro_jaccard) > 0:
            results['macro']['jaccard'] = np.mean(macro_jaccard)
            results['macro']['jaccard_std'] = np.std(macro_jaccard)
        if len(macro_hd95) > 0:
            results['macro']['hd95'] = np.mean(macro_hd95)
            results['macro']['hd95_std'] = np.std(macro_hd95)
        if len(macro_nsd) > 0:
            results['macro']['nsd'] = np.mean(macro_nsd)
            results['macro']['nsd_std'] = np.std(macro_nsd)
        if len(macro_biou) > 0:
            results['macro']['boundary_iou'] = np.mean(macro_biou)
            results['macro']['boundary_iou_std'] = np.std(macro_biou)
        
        return results


def save_checkpoint(
    state: Dict,
    is_best: bool,
    checkpoint_dir: str,
    filename: str = 'checkpoint.pth'
):
    """
    Save model checkpoint.
    
    Args:
        state: State dictionary with model, optimizer, etc.
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoints
        filename: Checkpoint filename
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model.pth')
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        strict: Whether to strictly enforce key matching
    
    Returns:
        State dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    state_dict = checkpoint.get('student', checkpoint.get('model', checkpoint.get('state_dict')))
    if state_dict is None:
        raise KeyError("Checkpoint must contain 'student', 'model', or 'state_dict'")
    model.load_state_dict(state_dict, strict=strict)
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint
