"""
Training script for HierMedKD.

Usage:
    python train.py --config configs/synapse.yaml --seed 2027
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.teacher import SwinUNETeacher
from models.student import LiteMedFormerS
from models.adapters import StageAdapters
from losses.__init__ import HierMedKDLoss
from losses.supervised_loss import SupervisedLoss
from data.synapse import SynapseDataset
from data.acdc import ACDCDataset
from data.isic import ISICDataset
from utils import save_checkpoint, SegmentationMetrics
from config_loader import load_config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(config, split):
    """Get dataset based on configuration."""
    dataset_name = config['data']['dataset_name']
    data_dir = config['data']['data_dir']
    image_size = tuple(config['data']['image_size'])
    in_channels = config['data']['in_channels']
    num_classes = config['data']['num_classes']
    ignore_label = config['data']['ignore_label']
    split_seed = config['data']['split']['seed']
    flip_probability = config['data'].get('horizontal_flip_probability', 0.0) if split == 'train' else 0.0
    
    if dataset_name == 'synapse':
        return SynapseDataset(
            data_dir, split, image_size, in_channels, num_classes, ignore_label,
            None, split_seed, flip_probability
        )
    elif dataset_name == 'acdc':
        return ACDCDataset(
            data_dir, split, image_size, in_channels, num_classes, ignore_label,
            None, split_seed, flip_probability
        )
    elif dataset_name == 'isic':
        return ISICDataset(
            data_dir, split, image_size, in_channels, num_classes, ignore_label,
            None, split_seed, flip_probability
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def train_one_epoch(
    teacher,
    student,
    adapters,
    train_loader,
    supervised_loss,
    kd_criterion,
    optimizer,
    scaler,
    device,
    config,
    epoch
):
    """Train for one epoch."""
    teacher.eval()
    student.train()
    adapters.train()
    
    total_loss = 0.0
    total_seg_loss = 0.0
    total_kd_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=scaler.is_enabled()):
            # Forward pass
            with torch.no_grad():
                teacher_out = teacher(images)
                teacher_logits = teacher_out['logits']
                teacher_features = teacher_out['features']
            
            student_out = student(images)
            student_logits = student_out['logits']
            student_features = student_out['features']
            
            # Align student features to teacher channels
            aligned_features = adapters(student_features)
            
            # Compute supervised loss
            seg_loss_dict = supervised_loss(student_logits, labels, config['data']['ignore_label'])
            seg_loss = seg_loss_dict['seg_loss']
            
            # Compute KD loss
            kd_loss_dict = kd_criterion(
                teacher_features,
                aligned_features,
                teacher_logits,
                student_logits,
                labels,
                config['data']['ignore_label']
            )
            kd_value = kd_loss_dict['kd_loss']
            
            # Total loss
            loss = seg_loss + kd_value
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        trainable_parameters = list(student.parameters())
        if adapters is not None:
            trainable_parameters += list(adapters.parameters())
        torch.nn.utils.clip_grad_norm_(trainable_parameters, config['training']['gradient_clip'])
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_kd_loss += kd_value.item()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'seg': f"{seg_loss.item():.4f}",
            'kd': f"{kd_value.item():.4f}"
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_seg_loss = total_seg_loss / len(train_loader)
    avg_kd_loss = total_kd_loss / len(train_loader)
    
    return avg_loss, avg_seg_loss, avg_kd_loss


def validate(
    student,
    val_loader,
    metrics,
    device,
    config
):
    """Validate the model."""
    student.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            with autocast(enabled=(config['training']['mixed_precision'] and device.type == 'cuda')):
                student_out = student(images)
                logits = student_out['logits']
            
            metrics.update(logits, labels)
    
    # Get metrics
    metric_results = metrics.get_metrics()
    
    return metric_results


def main():
    parser = argparse.ArgumentParser(description='Train HierMedKD')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--seed', type=int, default=2027, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument(
        '--teacher_checkpoint',
        type=str,
        required=True,
        help='Checkpoint for the trained Swin-style teacher (required for distillation)'
    )
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Override seed if provided
    if args.seed is not None:
        config['reproducibility']['seed'] = args.seed
    
    # Set seed
    set_seed(config['reproducibility']['seed'])
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = get_dataset(config, 'train')
    val_dataset = get_dataset(config, 'val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create models
    print("Creating models...")
    teacher = SwinUNETeacher(
        in_channels=config['data']['in_channels'],
        num_classes=config['data']['num_classes'],
        embed_dim=config['model']['teacher']['embed_dim'],
        depths=config['model']['teacher']['depths'],
        num_heads=config['model']['teacher']['num_heads'],
        window_size=config['model']['teacher']['window_size'],
        mlp_ratio=config['model']['teacher']['mlp_ratio']
    ).to(device)

    teacher_state = torch.load(args.teacher_checkpoint, map_location=device)
    if isinstance(teacher_state, dict):
        teacher_state = teacher_state.get('teacher', teacher_state.get('model', teacher_state.get('state_dict', teacher_state)))
    teacher.load_state_dict(teacher_state, strict=True)
    
    student = LiteMedFormerS(
        in_channels=config['data']['in_channels'],
        num_classes=config['data']['num_classes'],
        encoder_channels=config['model']['student']['encoder_channels'],
        decoder_channels=config['model']['student']['decoder_channels']
    ).to(device)
    
    # Create adapters
    if config['model']['adapters']['use_adapters']:
        student_channels = config['model']['student']['encoder_channels']
        teacher_channels = [config['model']['teacher']['embed_dim'] * (2 ** i) for i in range(4)]
        adapters = StageAdapters(
            student_channels,
            teacher_channels,
            config['model']['adapters']['adapter_type']
        ).to(device)
    else:
        adapters = None
    
    # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Loss functions
    supervised_loss = SupervisedLoss(epsilon=config['training']['kd']['epsilon'])
    kd_criterion = HierMedKDLoss(
        lambda_feat=config['training']['loss_weights']['lambda_feat'],
        lambda_att=config['training']['loss_weights']['lambda_att'],
        lambda_kl=config['training']['loss_weights']['lambda_kl'],
        temperature=config['training']['kd']['temperature'],
        beta=config['training']['kd']['beta'],
        kernel_size=config['training']['kd']['kernel_size'],
        epsilon=config['training']['kd']['epsilon'],
        confidence_gating=config['training']['kd']['confidence_gating']
    )
    
    # Optimizer
    params = list(student.parameters())
    if adapters is not None:
        params += list(adapters.parameters())
    
    optimizer = optim.AdamW(
        params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    if config['training']['cosine_scheduler']:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=1e-6
        )
    else:
        scheduler = None
    
    # Mixed precision
    amp_enabled = bool(config['training']['mixed_precision'] and device.type == 'cuda')
    scaler = GradScaler(enabled=amp_enabled)
    
    # Metrics
    metrics = SegmentationMetrics(
        num_classes=config['data']['num_classes'],
        ignore_label=config['data']['ignore_label'],
        spacing=(1.0, 1.0),
        nsd_tolerance=config['evaluation']['nsd_tolerance'],
        boundary_dilation_ratio=config['evaluation']['boundary_dilation_ratio']
    )
    
    # Tensorboard
    if config['logging']['tensorboard']:
        writer = SummaryWriter(config['logging']['log_dir'])
    else:
        writer = None
    
    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0.0
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        student.load_state_dict(checkpoint['student'])
        if adapters is not None and 'adapters' in checkpoint:
            adapters.load_state_dict(checkpoint['adapters'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_dice = checkpoint['best_dice']
    
    # Training loop
    print("Starting training...")
    num_epochs = config['training']['num_epochs']
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss, train_seg_loss, train_kd_loss = train_one_epoch(
            teacher, student, adapters, train_loader,
            supervised_loss, kd_criterion, optimizer, scaler, device, config, epoch + 1
        )
        
        # Validate
        metrics.reset()
        val_metrics = validate(student, val_loader, metrics, device, config)
        val_dice = val_metrics['macro']['dice']
        
        print(f"Train Loss: {train_loss:.4f}, Seg: {train_seg_loss:.4f}, KD: {train_kd_loss:.4f}")
        print(f"Val Dice: {val_dice:.4f}")
        
        # Logging
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/seg', train_seg_loss, epoch)
            writer.add_scalar('Loss/kd', train_kd_loss, epoch)
            writer.add_scalar('Metrics/dice', val_dice, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
        
        # Save checkpoint
        is_best = val_dice > best_dice
        best_dice = max(best_dice, val_dice)
        
        state = {
            'epoch': epoch + 1,
            'student': student.state_dict(),
            'adapters': adapters.state_dict() if adapters is not None else None,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'best_dice': best_dice,
            'config': config
        }
        
        checkpoint_dir = os.path.join(config['checkpoint']['save_dir'], config['data']['dataset_name'])
        save_checkpoint(state, is_best, checkpoint_dir)
        
        print(f"Best Dice: {best_dice:.4f}")
    
    print("Training completed!")
    
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
