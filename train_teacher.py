"""Train the independently implemented Swin-style teacher.

Usage:
    python train_teacher.py --config configs/synapse.yaml --seed 2027
"""

import argparse
import os

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_loader import load_config
from losses.supervised_loss import SupervisedLoss
from models.teacher import SwinUNETeacher
from train import get_dataset, set_seed, validate
from utils import SegmentationMetrics, save_checkpoint, split_manifest_sha256


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, ignore_label):
    model.train()
    running_loss = 0.0
    progress = tqdm(loader, desc="Teacher training")
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler.is_enabled()):
            logits = model(images)['logits']
            loss = criterion(logits, labels, ignore_label)['seg_loss']
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += float(loss.item())
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser(description="Train the HierMedKD teacher")
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config['reproducibility']['seed'] = args.seed
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    train_dataset = get_dataset(config, 'train')
    val_dataset = get_dataset(config, 'val')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
    )

    model = SwinUNETeacher(
        in_channels=config['data']['in_channels'],
        num_classes=config['data']['num_classes'],
        embed_dim=config['model']['teacher']['embed_dim'],
        depths=config['model']['teacher']['depths'],
        num_heads=config['model']['teacher']['num_heads'],
        window_size=config['model']['teacher']['window_size'],
        mlp_ratio=config['model']['teacher']['mlp_ratio'],
    ).to(device)
    criterion = SupervisedLoss(epsilon=config['training']['kd']['epsilon'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=1e-6,
    ) if config['training']['cosine_scheduler'] else None
    scaler = GradScaler(
        enabled=bool(config['training']['mixed_precision'] and device.type == 'cuda')
    )
    metrics = SegmentationMetrics(
        num_classes=config['data']['num_classes'],
        ignore_label=config['data']['ignore_label'],
        nsd_tolerance=config['evaluation']['nsd_tolerance'],
        boundary_dilation_ratio=config['evaluation']['boundary_dilation_ratio'],
    )

    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state['teacher'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None and state.get('scheduler') is not None:
            scheduler.load_state_dict(state['scheduler'])
        start_epoch = int(state['epoch'])
        best_dice = float(state['best_dice'])

    checkpoint_dir = os.path.join(
        config['checkpoint']['save_dir'],
        'teacher',
        config['data']['dataset_name'],
    )
    for epoch in range(start_epoch, config['training']['num_epochs']):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            config['data']['ignore_label'],
        )
        metrics.reset()
        validation = validate(model, val_loader, metrics, device, config)
        val_dice = float(validation['macro']['dice'])
        if scheduler is not None:
            scheduler.step()
        is_best = val_dice > best_dice
        best_dice = max(best_dice, val_dice)
        state = {
            'epoch': epoch + 1,
            'teacher': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'best_dice': best_dice,
            'config': config,
            'train_split_sha256': split_manifest_sha256(train_dataset),
            'validation_split_sha256': split_manifest_sha256(val_dataset),
        }
        save_checkpoint(state, is_best, checkpoint_dir)
        print(
            f"Epoch {epoch + 1}: loss={train_loss:.4f}, "
            f"val_dice={val_dice:.4f}, best={best_dice:.4f}"
        )


if __name__ == '__main__':
    main()
