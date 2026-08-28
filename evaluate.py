"""
Evaluation script for HierMedKD.

Usage:
    python evaluate.py --config configs/synapse.yaml --checkpoint checkpoints/synapse/best_model.pth --split test
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.student import LiteMedFormerS
from data.synapse import SynapseDataset
from data.acdc import ACDCDataset
from data.isic import ISICDataset
from utils import SegmentationMetrics, load_checkpoint
from config_loader import load_config


def get_dataset(config, split):
    """Get dataset based on configuration."""
    dataset_name = config['data']['dataset_name']
    data_dir = config['data']['data_dir']
    image_size = tuple(config['data']['image_size'])
    in_channels = config['data']['in_channels']
    num_classes = config['data']['num_classes']
    ignore_label = config['data']['ignore_label']
    split_seed = config['data']['split']['seed']
    
    if dataset_name == 'synapse':
        return SynapseDataset(
            data_dir, split, image_size, in_channels, num_classes, ignore_label, None, split_seed, 0.0
        )
    elif dataset_name == 'acdc':
        return ACDCDataset(
            data_dir, split, image_size, in_channels, num_classes, ignore_label, None, split_seed, 0.0
        )
    elif dataset_name == 'isic':
        return ISICDataset(
            data_dir, split, image_size, in_channels, num_classes, ignore_label, None, split_seed, 0.0
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def compute_confidence_interval(scores, confidence_level=0.95):
    """
    Compute confidence interval using paired bootstrap.
    
    Args:
        scores: List of scores
        confidence_level: Confidence level (default: 0.95)
    
    Returns:
        (mean, lower, upper) tuple
    """
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    
    mean = np.mean(scores)
    
    # Bootstrap for confidence interval
    n_bootstrap = config['evaluation']['num_bootstrap'] if 'config' in globals() else 1000
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    lower = np.percentile(bootstrap_means, (1 - confidence_level) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + confidence_level) / 2 * 100)
    
    return mean, lower, upper


def evaluate(
    model,
    data_loader,
    metrics,
    device,
    config,
    save_predictions=False,
    output_dir=None
):
    """
    Evaluate model on dataset.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation
        metrics: SegmentationMetrics instance
        device: Device to run on
        config: Configuration dictionary
        save_predictions: Whether to save predictions
        output_dir: Directory to save predictions
    
    Returns:
        Dictionary of evaluation results
    """
    model.eval()
    
    if save_predictions and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(data_loader, desc="Evaluating")):
            images = images.to(device)
            labels = labels.to(device)
            
            with autocast(enabled=(config['training']['mixed_precision'] and device.type == 'cuda')):
                model_out = model(images)
                logits = model_out['logits']
            
            # Update metrics
            metrics.update(logits, labels)
            
            # Save predictions if requested
            if save_predictions and output_dir:
                preds = logits.argmax(dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                for b in range(preds.shape[0]):
                    pred_path = os.path.join(output_dir, f"pred_{idx}_{b}.npy")
                    label_path = os.path.join(output_dir, f"label_{idx}_{b}.npy")
                    np.save(pred_path, preds[b])
                    np.save(label_path, labels_np[b])
    
    # Get metrics
    metric_results = metrics.get_metrics()
    
    # Compute confidence intervals
    results_with_ci = {}
    
    for class_name in metric_results:
        if class_name == 'macro':
            continue
        
        results_with_ci[class_name] = {}
        
        if 'dice' in metric_results[class_name]:
            # Per-case scores would need to be tracked separately
            # For now, just report mean
            results_with_ci[class_name]['dice'] = metric_results[class_name]['dice']
            results_with_ci[class_name]['dice_std'] = metric_results[class_name]['dice_std']

        if 'jaccard' in metric_results[class_name]:
            results_with_ci[class_name]['jaccard'] = metric_results[class_name]['jaccard']
            results_with_ci[class_name]['jaccard_std'] = metric_results[class_name]['jaccard_std']
        
        if 'hd95' in metric_results[class_name]:
            results_with_ci[class_name]['hd95'] = metric_results[class_name]['hd95']
            results_with_ci[class_name]['hd95_std'] = metric_results[class_name]['hd95_std']
        
        if 'nsd' in metric_results[class_name]:
            results_with_ci[class_name]['nsd'] = metric_results[class_name]['nsd']
            results_with_ci[class_name]['nsd_std'] = metric_results[class_name]['nsd_std']
        
        if 'boundary_iou' in metric_results[class_name]:
            results_with_ci[class_name]['boundary_iou'] = metric_results[class_name]['boundary_iou']
            results_with_ci[class_name]['boundary_iou_std'] = metric_results[class_name]['boundary_iou_std']
    
    # Macro averages
    results_with_ci['macro'] = metric_results['macro']
    
    return results_with_ci


def main():
    parser = argparse.ArgumentParser(description='Evaluate HierMedKD')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to evaluate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--save_predictions', action='store_true', help='Save prediction masks')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Directory to save predictions')
    
    args = parser.parse_args()
    
    global config
    config = load_config(args.config)
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print(f"Loading {args.split} dataset...")
    dataset = get_dataset(config, args.split)
    data_loader = DataLoader(
        dataset,
        batch_size=1,  # Use batch size 1 for evaluation
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Samples: {len(dataset)}")
    
    # Create model
    print("Creating model...")
    model = LiteMedFormerS(
        in_channels=config['data']['in_channels'],
        num_classes=config['data']['num_classes'],
        encoder_channels=config['model']['student']['encoder_channels'],
        decoder_channels=config['model']['student']['decoder_channels']
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, model, strict=True)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Metrics
    metrics = SegmentationMetrics(
        num_classes=config['data']['num_classes'],
        ignore_label=config['data']['ignore_label'],
        spacing=(1.0, 1.0),
        nsd_tolerance=config['evaluation']['nsd_tolerance'],
        boundary_dilation_ratio=config['evaluation']['boundary_dilation_ratio']
    )
    
    # Evaluate
    print("Evaluating...")
    results = evaluate(
        model,
        data_loader,
        metrics,
        device,
        config,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print("\nPer-class results:")
    for class_name in results:
        if class_name == 'macro':
            continue
        print(f"\n{class_name}:")
        if 'dice' in results[class_name]:
            print(f"  Dice: {results[class_name]['dice']:.4f} ± {results[class_name]['dice_std']:.4f}")
        if 'jaccard' in results[class_name]:
            print(f"  Jaccard: {results[class_name]['jaccard']:.4f} ± {results[class_name]['jaccard_std']:.4f}")
        if 'hd95' in results[class_name]:
            print(f"  HD95: {results[class_name]['hd95']:.4f} ± {results[class_name]['hd95_std']:.4f}")
        if 'nsd' in results[class_name]:
            print(f"  NSD: {results[class_name]['nsd']:.4f} ± {results[class_name]['nsd_std']:.4f}")
        if 'boundary_iou' in results[class_name]:
            print(f"  Boundary IoU: {results[class_name]['boundary_iou']:.4f} ± {results[class_name]['boundary_iou_std']:.4f}")
    
    print("\nMacro average:")
    macro = results['macro']
    if 'dice' in macro:
        print(f"  Dice: {macro['dice']:.4f} ± {macro['dice_std']:.4f}")
    if 'jaccard' in macro:
        print(f"  Jaccard: {macro['jaccard']:.4f} ± {macro['jaccard_std']:.4f}")
    if 'hd95' in macro:
        print(f"  HD95: {macro['hd95']:.4f} ± {macro['hd95_std']:.4f}")
    if 'nsd' in macro:
        print(f"  NSD: {macro['nsd']:.4f} ± {macro['nsd_std']:.4f}")
    if 'boundary_iou' in macro:
        print(f"  Boundary IoU: {macro['boundary_iou']:.4f} ± {macro['boundary_iou_std']:.4f}")
    
    print("=" * 60)
    
    # Save results to JSON
    results_path = os.path.join(args.output_dir, 'results.json') if args.save_predictions else 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
