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
from utils import (
    SegmentationMetrics,
    compute_boundary_iou,
    compute_dice,
    compute_hd95,
    compute_jaccard,
    compute_nsd,
    load_checkpoint,
    sha256_file,
    split_manifest_sha256,
)
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


def compute_confidence_interval(
    scores,
    confidence_level=0.95,
    n_bootstrap=10000,
    random_seed=2027,
):
    """
    Compute confidence interval using paired bootstrap.

    Args:
        scores: List of scores
        confidence_level: Confidence level (default: 0.95)

    Returns:
        (mean, lower, upper) tuple
    """
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float('nan'), float('nan'), float('nan')

    mean = float(np.mean(scores))
    rng = np.random.RandomState(random_seed)
    indices = rng.randint(0, scores.size, size=(int(n_bootstrap), scores.size))
    bootstrap_means = scores[indices].mean(axis=1)
    lower = np.percentile(bootstrap_means, (1 - confidence_level) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + confidence_level) / 2 * 100)

    return mean, float(lower), float(upper)


def _restore_mask(mask, native_shape):
    """Nearest-neighbor restoration of a categorical mask to its native grid."""
    from skimage.transform import resize

    return resize(
        mask,
        tuple(int(v) for v in native_shape),
        order=0,
        mode='edge',
        anti_aliasing=False,
        preserve_range=True,
    ).astype(np.int64)


def _metric_record(pred, label, spacing, config):
    """Compute one native-grid volume/image record."""
    num_classes = int(config['data']['num_classes'])
    ignore_label = int(config['data']['ignore_label'])
    record = {}

    for class_id in range(1, num_classes):
        pred_mask = (pred == class_id).astype(np.uint8)
        label_mask = (label == class_id).astype(np.uint8)
        if not label_mask.any():
            continue

        class_record = {
            'dice': compute_dice(
                torch.from_numpy(pred_mask),
                torch.from_numpy(label_mask),
                ignore_label,
            ),
            'jaccard': compute_jaccard(
                torch.from_numpy(pred_mask),
                torch.from_numpy(label_mask),
                ignore_label,
            ),
            'hd95': compute_hd95(pred_mask, label_mask, spacing, ignore_label),
            'nsd': compute_nsd(
                pred_mask,
                label_mask,
                float(config['evaluation']['nsd_tolerance']),
                spacing,
                ignore_label,
            ),
        }
        if pred.ndim == 2:
            class_record['boundary_iou'] = compute_boundary_iou(
                pred_mask,
                label_mask,
                float(config['evaluation']['boundary_dilation_ratio']),
                ignore_label,
            )
        record[f'class_{class_id}'] = class_record

    metric_names = ('dice', 'jaccard', 'hd95', 'nsd', 'boundary_iou')
    record['macro'] = {}
    for metric_name in metric_names:
        values = [
            values[metric_name]
            for key, values in record.items()
            if key.startswith('class_') and metric_name in values
        ]
        if values:
            record['macro'][metric_name] = float(np.mean(values))
    return record


def _aggregate_patient_records(volume_records, config):
    """Average phases within a patient, then bootstrap patients/images."""
    patient_values = {}
    for volume_record in volume_records:
        case_id = volume_record['case_id']
        patient = patient_values.setdefault(case_id, {})
        for group_name, group_values in volume_record['metrics'].items():
            group = patient.setdefault(group_name, {})
            for metric_name, value in group_values.items():
                group.setdefault(metric_name, []).append(float(value))

    case_scores = {}
    for case_id, groups in patient_values.items():
        case_scores[case_id] = {
            group_name: {
                metric_name: float(np.mean(values))
                for metric_name, values in metrics.items()
            }
            for group_name, metrics in groups.items()
        }

    groups = sorted({group for case in case_scores.values() for group in case})
    results = {}
    n_bootstrap = int(config['evaluation'].get('num_bootstrap', 10000))
    confidence_level = float(config['evaluation'].get('confidence_level', 0.95))
    seed = int(config.get('reproducibility', {}).get('seed', 2027))

    for group_name in groups:
        metric_names = sorted({
            metric
            for case in case_scores.values()
            for metric in case.get(group_name, {})
        })
        results[group_name] = {}
        for metric_name in metric_names:
            values = [
                case[group_name][metric_name]
                for case in case_scores.values()
                if metric_name in case.get(group_name, {})
            ]
            mean, lower, upper = compute_confidence_interval(
                values,
                confidence_level=confidence_level,
                n_bootstrap=n_bootstrap,
                random_seed=seed,
            )
            results[group_name][metric_name] = mean
            results[group_name][f'{metric_name}_std'] = float(np.std(values))
            results[group_name][f'{metric_name}_ci_low'] = lower
            results[group_name][f'{metric_name}_ci_high'] = upper
            results[group_name][f'{metric_name}_ci_half_width'] = float(
                max(mean - lower, upper - mean)
            )

    results['case_scores'] = case_scores
    return results


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
    dataset = data_loader.dataset
    native_evaluation = all(
        hasattr(dataset, name)
        for name in ('sample_metadata', 'native_label_slice')
    )
    volume_buffers = {}

    if save_predictions and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(data_loader, desc="Evaluating")):
            images = images.to(device)
            labels = labels.to(device)

            with autocast(enabled=(config['training']['mixed_precision'] and device.type == 'cuda')):
                model_out = model(images)
                logits = model_out['logits']

            if native_evaluation:
                if images.shape[0] != 1:
                    raise ValueError("Native-grid evaluation requires batch_size=1")
                metadata = dataset.sample_metadata(idx)
                pred_224 = logits.argmax(dim=1)[0].cpu().numpy()
                native_label = dataset.native_label_slice(idx)
                native_pred = _restore_mask(pred_224, native_label.shape)
                volume_id = metadata['volume_id']

                if metadata['slice_index'] is None:
                    volume_buffers[volume_id] = {
                        'case_id': metadata['case_id'],
                        'spacing': tuple(metadata['spacing']),
                        'pred': native_pred,
                        'label': native_label,
                    }
                else:
                    if volume_id not in volume_buffers:
                        native_shape = tuple(metadata['native_shape'])
                        volume_buffers[volume_id] = {
                            'case_id': metadata['case_id'],
                            'spacing': tuple(metadata['spacing']),
                            'pred': np.zeros(native_shape, dtype=np.int64),
                            'label': np.zeros(native_shape, dtype=np.int64),
                        }
                    slice_index = int(metadata['slice_index'])
                    volume_buffers[volume_id]['pred'][:, :, slice_index] = native_pred
                    volume_buffers[volume_id]['label'][:, :, slice_index] = native_label
            else:
                metrics.update(logits, labels)

    if not native_evaluation:
        return metrics.get_metrics()

    volume_records = []
    for volume_id, volume in sorted(volume_buffers.items()):
        metric_record = _metric_record(
            volume['pred'], volume['label'], volume['spacing'], config
        )
        volume_records.append({
            'case_id': volume['case_id'],
            'volume_id': volume_id,
            'metrics': metric_record,
        })
        if save_predictions and output_dir:
            safe_name = volume_id.replace('/', '__').replace('\\', '__')
            np.save(os.path.join(output_dir, f'{safe_name}_pred.npy'), volume['pred'])
            np.save(os.path.join(output_dir, f'{safe_name}_label.npy'), volume['label'])

    return _aggregate_patient_records(volume_records, config)


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
    results['_metadata'] = {
        'checkpoint_sha256': sha256_file(args.checkpoint),
        'split_sha256': split_manifest_sha256(dataset),
        'split': args.split,
        'dataset': config['data']['dataset_name'],
        'seed': config['reproducibility']['seed'],
    }

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\nPer-class results:")
    for class_name in results:
        if class_name in ('macro', 'case_scores', '_metadata'):
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
