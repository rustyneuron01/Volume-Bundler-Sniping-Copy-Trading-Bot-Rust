#!/usr/bin/env python3
"""
Train ReStraV-style detector: frozen DINOv2 + 21-D geometry + MLP.

Uses the same CSVs and .npy (or .npy.zst) data as dinov2. Expects video shape (24, 518, 518, 3);
model resizes to 224×224 internally.

Pipeline:
  1. Calibration pass: sample N train videos, compute 21-D features, set mean/std in model.
  2. Train only the MLP (classifier) with CrossEntropyLoss.

Usage (single GPU):
  python train_restrav.py \\
    --train-csv ../dinov2/train.csv \\
    --val-csv ../dinov2/val_eval.csv \\
    --data-root /path/to/data \\
    --output-dir checkpoints/restrav \\
    [--compressed for .npy.zst]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Load restrav model from file so dinov2 can keep using its own model
_ALT = Path(__file__).resolve().parent
_DINOV2 = _ALT.parent / "dinov2"
import importlib.util
_spec = importlib.util.spec_from_file_location("restrav_model", _ALT / "model.py")
_restrav_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_restrav_module)
ReStraVDetector = _restrav_module.ReStraVDetector
create_model = _restrav_module.create_model
NUM_GEOM_FEATURES = _restrav_module.NUM_GEOM_FEATURES

sys.path.insert(0, str(_DINOV2))
import train_dinov2 as _dd
VideoNpyDataset = _dd.VideoNpyDataset
EXPECTED_SHAPE = _dd.EXPECTED_SHAPE


def parse_args():
    p = argparse.ArgumentParser(description="Train ReStraV-style detector")
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--val-csv", type=str, required=True)
    p.add_argument("--data-root", type=str, default=None,
                   help="Root for npy paths in CSV (e.g. rclone mount point: /mnt/b2data)")
    p.add_argument("--output-dir", type=str, default="checkpoints/restrav")
    p.add_argument("--mmap", action="store_true",
                   help="Use mmap for .npy (saves RAM on mount; can increase read traffic)")
    p.add_argument("--compressed", action="store_true", help="Use .npy.zst with decoded cache")
    p.add_argument("--decoded-root", type=str, default=None,
                   help="Cache for decoded .npy.zst (use local SSD; default <data_root>/_decoded_step_cache)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--calib-samples", type=int, default=20000,
                   help="Number of training samples for 21-D mean/std calibration")
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers (0=auto: max(4, min(16, cpus))), load+aug in parallel with GPU")
    p.add_argument("--prefetch-factor", type=int, default=4,
                   help="Batches prefetched per worker (like dinov2)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", type=str, default="dinov2_vits14")
    # Optional augmentations (in DataLoader workers, parallel with train). Default 0 to keep geometry clean.
    p.add_argument("--hflip-prob", type=float, default=0.0)
    p.add_argument("--color-jitter", type=float, default=0.0)
    p.add_argument("--color-jitter-prob", type=float, default=0.0)
    p.add_argument("--jpeg-prob", type=float, default=0.0)
    p.add_argument("--jpeg-quality-min", type=int, default=30)
    p.add_argument("--jpeg-quality-max", type=int, default=95)
    p.add_argument("--gaussian-noise-prob", type=float, default=0.0)
    p.add_argument("--gaussian-noise-std-min", type=float, default=0.005)
    p.add_argument("--gaussian-noise-std-max", type=float, default=0.03)
    p.add_argument("--gaussian-blur-prob", type=float, default=0.0)
    p.add_argument("--resize-quality-prob", type=float, default=0.0)
    return p.parse_args()


def calibrate_feature_normalization(model, loader, device, max_samples):
    """Run model over train samples, collect 21-D features, set mean/std in model."""
    model.eval()
    feats = []
    n = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Calibrating 21-D", leave=False):
            videos = batch[0].to(device, non_blocking=True).float()
            if videos.max() <= 1.0:
                videos = videos * 255.0
            B = videos.shape[0]
            feat_21 = model.get_geometry_features(videos)
            feats.append(feat_21.cpu().numpy())
            n += B
            if n >= max_samples:
                break
    feats = np.vstack(feats)[:max_samples]
    mean = feats.mean(axis=0).astype(np.float32)
    std = feats.std(axis=0).astype(np.float32) + 1e-8
    model.set_feature_normalization(mean, std)
    return mean, std


def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    brier_sum = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            videos = batch[0].to(device, non_blocking=True).float()
            if videos.max() <= 1.0:
                videos = videos * 255.0
            labels = batch[1].to(device, non_blocking=True)
            logits = model(videos)
            probs = F.softmax(logits.float(), dim=1)
            preds = probs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            p_real = probs[:, 1]
            brier_sum += ((p_real - labels.float()) ** 2).sum().item()
    acc = correct / total if total else 0.0
    brier = brier_sum / total if total else 0.0
    return acc, brier


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Data
    decoded_root = Path(args.decoded_root) if args.decoded_root else None
    if args.compressed and decoded_root is None and args.data_root:
        decoded_root = Path(args.data_root) / "_decoded_step_cache"
    train_ds = VideoNpyDataset(
        args.train_csv,
        data_root=args.data_root,
        mmap=args.mmap,
        compressed=args.compressed,
        decoded_root=decoded_root,
    )
    # Augmentations run on CPU in DataLoader workers (parallel with GPU), like dinov2. Default 0 to keep geometry clean.
    train_ds.set_aug_params(
        hflip_prob=args.hflip_prob,
        color_jitter=args.color_jitter,
        color_jitter_prob=args.color_jitter_prob,
        jpeg_prob=args.jpeg_prob,
        jpeg_quality_min=args.jpeg_quality_min,
        jpeg_quality_max=args.jpeg_quality_max,
        gaussian_noise_prob=args.gaussian_noise_prob,
        gaussian_noise_std_min=args.gaussian_noise_std_min,
        gaussian_noise_std_max=args.gaussian_noise_std_max,
        gaussian_blur_prob=args.gaussian_blur_prob,
        resize_quality_prob=args.resize_quality_prob,
    )
    val_ds = VideoNpyDataset(
        args.val_csv,
        data_root=args.data_root,
        mmap=args.mmap,
        compressed=args.compressed,
        decoded_root=decoded_root,
    )
    val_ds.set_aug_params(hflip_prob=0.0, color_jitter=0.0, color_jitter_prob=0.0)

    # Like dinov2: workers do load + decode + augment in parallel; prefetch overlaps with GPU train
    num_workers = args.workers if args.workers > 0 else max(4, min(16, (os.cpu_count() or 8)))
    prefetch = args.prefetch_factor if num_workers > 0 else None
    persistent = num_workers > 0
    print(f"  [loader] num_workers={num_workers}, prefetch_factor={args.prefetch_factor} (load+aug || train)")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=prefetch,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch,
        persistent_workers=persistent,
    )

    total_train = len(train_ds)
    batches_per_epoch = total_train // args.batch_size
    total_batches = batches_per_epoch * args.epochs
    # ViT-S 224: ~32*24=768 images/batch; ~0.2–0.5 s/batch on V100/A100 (backbone is bottleneck)
    est_sec_per_batch = 0.35
    calib_batches = (args.calib_samples + args.batch_size - 1) // args.batch_size
    est_min = (total_batches * est_sec_per_batch + calib_batches * est_sec_per_batch) / 60
    print(f"  [time] train samples={total_train:,}  batches/epoch={batches_per_epoch:,}  total_batches={total_batches:,}")
    print(f"  [time] est. ~{est_min:.0f} min (calib + train; GPU-dependent, ~{est_sec_per_batch}s/batch for ViT-S 224)")

    # Model
    model = create_model(backbone=args.backbone, pretrained=True).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.classifier.parameters())
    print(f"Total params: {total_p/1e6:.2f}M  Trainable (MLP only): {trainable_p/1e3:.1f}K")

    # Calibration: compute 21-D mean/std from training data
    print("Calibrating 21-D feature normalization...")
    mean, std = calibrate_feature_normalization(
        model, train_loader, device, max_samples=args.calib_samples
    )
    print(f"  feat_mean range [{mean.min():.4f}, {mean.max():.4f}]  feat_std min {std.min():.4f}")

    # Optimizer only for classifier
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            videos = batch[0].to(device, non_blocking=True).float()
            if videos.max() <= 1.0:
                videos = videos * 255.0
            labels = batch[1].to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(videos)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)
        acc, brier = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}  train_loss={train_loss:.4f}  val_acc={acc:.4f}  val_brier={brier:.4f}")
        if acc > best_acc:
            best_acc = acc
            ckpt = {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_acc": acc,
                "val_brier": brier,
            }
            torch.save(ckpt, Path(args.output_dir) / "best.pt")
            print(f"  -> saved best.pt (acc={acc:.4f})")
        torch.save(
            {"state_dict": model.state_dict(), "epoch": epoch + 1},
            Path(args.output_dir) / "latest.pt",
        )

    elapsed_min = (time.perf_counter() - t0) / 60
    print(f"Done. Best val_acc={best_acc:.4f}. Checkpoints in {args.output_dir}/")
    print(f"Elapsed: {elapsed_min:.1f} min")
    print("Export for gasbench: save model.state_dict() to model.safetensors (see export script).")


if __name__ == "__main__":
    main()
