#!/usr/bin/env python3
"""
VideoMAE v2 / MViTv2 Streaming Trainer - High SN34 focus

Key upgrades:
- VideoMAE v2 ViT-L fine-tuning
- In-model (T, b) calibration
- Optional MViTv2-Base ensemble evaluation
- Strict BGR->RGB conversion before any model normalization
"""

import os
import sys
import shutil
import subprocess
import threading
import time
import argparse
import logging
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import timedelta
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm

# Environment optimizations for 4x H200
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["CUDNN_BENCHMARK"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "0"
os.environ["NCCL_P2P_LEVEL"] = "NVL"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"

# Import models
sys.path.insert(0, os.path.dirname(__file__))
from models.video_mae_v2 import create_videomae_v2_vitl
from models.mvit_v2 import create_mvit_v2_base
from models.swin_l import create_swin_l
from models.swin3d_l import create_swin3d_l
from models.focal_loss import FocalLoss

# Directories
B2_BUCKET = "b2:sn33-bucket"
LOCAL_CACHE = os.path.expanduser("~/bittensor/epoch_cache")
MAX_CACHE_STEPS = 5  # Keep max 5 steps cached (current + 4 prefetch)
SAMPLES_PER_STEP = 100000  # 100K samples per step (download chunk)


# ===========================================
# DDP SETUP
# ===========================================
def setup_ddp():
    """Initialize DDP if running with torchrun."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(hours=2),
        )

        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    """Clean up DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ===========================================
# BACKBLAZE DATA MANAGER
# ===========================================
class BackblazeDataManager:
    """
    Manages downloading and caching of NPY files from Backblaze with background prefetch.
    """

    def __init__(self, b2_bucket: str = B2_BUCKET, local_cache: str = LOCAL_CACHE, rank: int = 0):
        self.b2_bucket = b2_bucket
        self.local_cache = Path(local_cache)
        self.local_cache.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.prefetch_thread = None
        self.prefetch_step = None
        self.step_files = defaultdict(list)

        if rank == 0:
            print("BackblazeDataManager initialized:")
            print(f"  Local cache: {self.local_cache}")
            print(f"  Max cached steps: {MAX_CACHE_STEPS}")
            print(f"  Samples per step: {SAMPLES_PER_STEP:,}")
            print(f"  Testing connection to {self.b2_bucket}...", flush=True)

            test_cmd = ["rclone", "lsd", self.b2_bucket, "--max-depth", "1"]
            test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
            if test_result.returncode == 0:
                print("  ✓ Connection OK", flush=True)
            else:
                print("  ⚠️  Connection test failed", flush=True)

    def download_files_batch(self, file_paths: List[str]) -> Tuple[int, int]:
        """Download files using rclone --files-from (batch download)."""
        import tempfile

        to_download = []
        skipped = 0
        for f in file_paths:
            local_path = self.local_cache / f
            if local_path.exists():
                skipped += 1
            else:
                to_download.append(f)
                local_path.parent.mkdir(parents=True, exist_ok=True)

        if not to_download:
            return (0, skipped)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
            for f in to_download:
                tf.write(f + "\n")
            files_from_path = tf.name

        try:
            cmd = [
                "rclone",
                "copy",
                self.b2_bucket,
                str(self.local_cache),
                "--files-from",
                files_from_path,
                "--transfers",
                "32",
                "--checkers",
                "64",
                "--multi-thread-streams",
                "4",
                "--fast-list",
                "--buffer-size",
                "64M",
                "--no-traverse",
                "--quiet",
            ]
            subprocess.run(cmd, capture_output=True, timeout=1800)
            downloaded = sum(1 for f in to_download if (self.local_cache / f).exists())
            return (downloaded, skipped)
        except subprocess.TimeoutExpired:
            print("  ⚠️  Timeout during download", flush=True)
            downloaded = sum(1 for f in to_download if (self.local_cache / f).exists())
            return (downloaded, skipped)
        finally:
            try:
                os.unlink(files_from_path)
            except Exception:
                pass

    def download_step_files(self, file_list: List[str], step: int) -> None:
        """Download files for a step (only rank 0)."""
        if self.rank != 0:
            return

        print(f"\n📥 Downloading step {step}: {len(file_list):,} files...", flush=True)

        existing = sum(1 for f in file_list if (self.local_cache / f).exists())
        if existing > 0:
            print(f"  ✓ {existing:,} files already cached", flush=True)

        dataset_files = defaultdict(list)
        for f in file_list:
            parts = f.split("/")
            dataset = parts[0] if len(parts) >= 2 else "unknown"
            dataset_files[dataset].append(f)

        total_downloaded = 0
        total_skipped = 0
        for dataset, files in dataset_files.items():
            downloaded, skipped = self.download_files_batch(files)
            total_downloaded += downloaded
            total_skipped += skipped
            if downloaded > 0:
                print(f"  ✓ {dataset}: {downloaded} downloaded", flush=True)

        if step >= 0:
            self.step_files[step] = [self.local_cache / f for f in file_list]

        print(f"  ✅ Step {step}: {total_downloaded} downloaded, {total_skipped} cached\n", flush=True)

    def prefetch_step_background(self, file_list: List[str], step: int) -> None:
        """Start background download of next step."""
        if self.rank != 0:
            return

        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            print(f"  ⏳ Waiting for prefetch step {self.prefetch_step} to complete...", flush=True)
            self.prefetch_thread.join()

        print(f"  🔄 Starting background prefetch for step {step}...", flush=True)
        self.prefetch_thread = threading.Thread(
            target=self.download_step_files,
            args=(file_list, step),
            daemon=True,
        )
        self.prefetch_step = step
        self.prefetch_thread.start()

    def wait_for_prefetch(self) -> None:
        """Wait for background prefetch to complete."""
        if self.rank != 0:
            return

        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            print(f"  ⏳ Waiting for prefetch step {self.prefetch_step}...", flush=True)
            self.prefetch_thread.join()
            print(f"  ✅ Prefetch step {self.prefetch_step} complete!", flush=True)

    def cleanup_old_steps(self, current_step: int) -> None:
        """Delete steps older than MAX_CACHE_STEPS (only rank 0)."""
        if self.rank != 0:
            return

        min_keep_step = max(0, current_step - MAX_CACHE_STEPS + 1)
        steps_to_delete = [s for s in list(self.step_files.keys()) if s < min_keep_step]

        for step in steps_to_delete:
            for path in self.step_files[step]:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
            del self.step_files[step]


# ===========================================
# DATASET
# ===========================================
class StreamingVideoDataset(Dataset):
    """Streaming dataset that loads from CSV sequentially."""

    def __init__(
        self,
        csv_path: str,
        data_manager: BackblazeDataManager,
        df: Optional[pd.DataFrame] = None,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        step: Optional[int] = None,
        verbose: bool = True,
        training: bool = True,
    ):
        self.csv_path = csv_path
        self.data_manager = data_manager
        self.step = step
        self.training = training

        if df is None:
            if verbose:
                if step is not None:
                    print(f"Loading CSV step {step}: {csv_path}", flush=True)
                else:
                    print(f"Loading CSV: {csv_path}", flush=True)

            self.df = pd.read_csv(csv_path)
            if end_idx is not None:
                self.df = self.df.iloc[start_idx:end_idx]
            else:
                self.df = self.df.iloc[start_idx:]
        else:
            self.df = df

        if verbose:
            print(f"  Dataset size: {len(self.df):,} samples", flush=True)
            print(f"  Real: {(self.df['label'] == 0).sum():,}", flush=True)
            print(f"  Fake: {(self.df['label'] == 1).sum():,}", flush=True)
            dataset_counts = self.df["dataset"].value_counts()
            print(f"  Datasets: {len(dataset_counts)}", flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bgr_path = row["bgr_path"]
        label = row["label"]
        dataset_name = row["dataset"]
        bgr_path = self.data_manager.local_cache / bgr_path

        try:
            video_bgr = np.load(bgr_path)  # [16, 224, 224, 3] uint8
            video = torch.from_numpy(video_bgr).float()
            video = video.permute(0, 3, 1, 2)  # [T, 3, H, W] BGR

            if self.training:
                if torch.rand(1).item() < 0.5:
                    video = torch.flip(video, dims=[-1])

            return video, label, dataset_name
        except Exception as exc:
            print(f"Error loading {bgr_path}: {exc}", flush=True)
            return torch.zeros(16, 3, 224, 224), label, dataset_name


# ===========================================
# EVALUATION METRICS (GasBench-compatible SN34)
# ===========================================
class Metrics:
    """Metrics computation matching GasBench SN34 scoring."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.all_probs = []
        self.all_labels = []
        self.dataset_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    def update(self, predictions, labels, probabilities, dataset_names):
        preds = predictions.cpu().numpy()
        labs = labels.cpu().numpy()
        probs = probabilities.cpu().numpy()

        self.tp += ((preds == 1) & (labs == 1)).sum()
        self.tn += ((preds == 0) & (labs == 0)).sum()
        self.fp += ((preds == 1) & (labs == 0)).sum()
        self.fn += ((preds == 0) & (labs == 1)).sum()

        self.all_probs.extend(probs[:, 1].tolist())
        self.all_labels.extend(labs.tolist())

        for pred, true, dataset in zip(preds, labs, dataset_names):
            self.dataset_stats[dataset]["total"] += 1
            if pred == true:
                self.dataset_stats[dataset]["correct"] += 1

    def compute(self, sn34_alpha: float = 1.2, sn34_beta: float = 1.8) -> Dict[str, float]:
        total = self.tp + self.tn + self.fp + self.fn
        accuracy = (self.tp + self.tn) / max(total, 1)

        numerator = (self.tp * self.tn) - (self.fp * self.fn)
        denominator = np.sqrt(
            (self.tp + self.fp)
            * (self.tp + self.fn)
            * (self.tn + self.fp)
            * (self.tn + self.fn)
        )
        mcc = numerator / max(denominator, 1e-8)

        probs = np.array(self.all_probs)
        labels = np.array(self.all_labels)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        bce = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        
        # Brier score (new SN34 metric)
        brier = np.mean((probs - labels) ** 2)

        # Updated SN34 formula (MCC + Brier with exponents)
        mcc_score = max(0.0, min((mcc + 1.0) / 2.0, 1.0)) ** sn34_alpha
        brier_score = max(0.0, (0.25 - brier) / 0.25) ** sn34_beta
        sn34 = math.sqrt(max(1e-12, mcc_score * brier_score))
        
        # Keep old CE score for logging
        def safe_exp_neg(x):
            x = max(0.0, min(x, 20.0))
            return math.exp(-x)
        base = 0.5
        val = safe_exp_neg(bce)
        ce_score = max(0.0, min((val - base) / (1.0 - base), 1.0))

        sn34_old = math.sqrt(max(1e-12, ((mcc + 1.0) / 2.0) * ce_score))

        real_acc = 100 * self.tn / max(self.tn + self.fp, 1)
        fake_acc = 100 * self.tp / max(self.tp + self.fn, 1)
        balance_gap = abs(real_acc - fake_acc)
        fn_rate = 100 * self.fn / max(self.tp + self.fn, 1)

        dataset_accs = {
            name: 100 * stats["correct"] / stats["total"]
            for name, stats in self.dataset_stats.items()
            if stats["total"] > 0
        }

        return {
            "accuracy": accuracy * 100,
            "real_accuracy": real_acc,
            "fake_accuracy": fake_acc,
            "balance_gap": balance_gap,
            "fn_rate": fn_rate,
            "mcc": mcc,
            "bce": bce,
            "brier": brier,
            "mcc_score": mcc_score,
            "ce_score": ce_score,
            "brier_score": brier_score,
            "sn34": sn34,
            "sn34_old": sn34_old,
            "dataset_accuracies": dataset_accs,
        }


# ===========================================
# TRAINER
# ===========================================
class ValidatorInputWrapper(nn.Module):
    """Accepts validator input: BGR float32 [B,T,3,H,W] in 0-255."""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * (1.0 / 255.0)
        x = x[:, :, [2, 1, 0], :, :]  # BGR -> RGB
        return self.core(x)


def build_model(args):
    if args.model == "videomae_v2":
        core = create_videomae_v2_vitl(
            pretrained=args.pretrained,
            num_classes=2,
            model_id=args.videomae_model_id,
            init_temperature=args.init_temperature,
            init_bias=args.init_bias,
        )
    elif args.model == "mvit_v2":
        core = create_mvit_v2_base(
            pretrained=args.pretrained,
            num_classes=2,
            init_temperature=args.init_temperature,
            init_bias=args.init_bias,
        )
    elif args.model == "swin_l":
        core = create_swin_l(
            pretrained=args.pretrained,
            num_classes=2,
            init_temperature=args.init_temperature,
            init_bias=args.init_bias,
        )
    elif args.model == "swin3d_l":
        core = create_swin3d_l(
            pretrained=args.pretrained,
            num_classes=2,
            init_temperature=args.init_temperature,
            init_bias=args.init_bias,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    return ValidatorInputWrapper(core)


def load_checkpoint_model(model_type: str, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if model_type == "videomae_v2":
        model_id = checkpoint.get("videomae_model_id", "MCG-NJU/videomae-v2-large")
        core = create_videomae_v2_vitl(pretrained=False, num_classes=2, model_id=model_id)
    elif model_type == "mvit_v2":
        core = create_mvit_v2_base(pretrained=False, num_classes=2)
    elif model_type == "swin_l":
        core = create_swin_l(pretrained=False, num_classes=2)
    elif model_type == "swin3d_l":
        core = create_swin3d_l(pretrained=False, num_classes=2)
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_type}")
    model = ValidatorInputWrapper(core)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate(model, dataloader, device, ensemble_model=None):
    model.eval()
    if ensemble_model is not None:
        ensemble_model.eval()

    metrics = Metrics()
    with torch.no_grad():
        for videos, labels, dataset_names in tqdm(dataloader, desc="Evaluating", leave=False):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(videos)
                if ensemble_model is not None:
                    ensemble_logits = ensemble_model(videos)
                    logits = (logits + ensemble_logits) / 2.0

            probabilities = F.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)
            metrics.update(predictions, labels, probabilities, dataset_names)

    return metrics.compute()


def collect_logits_and_labels(model, dataloader, device):
    """Collect raw logits and labels for calibration."""
    model.eval()
    calibrator = None
    core = model.core if hasattr(model, "core") else model
    if hasattr(core, "calibrator"):
        calibrator = core.calibrator
        core.calibrator = nn.Identity()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for videos, labels, _ in tqdm(dataloader, desc="Collecting logits"):
            videos = videos.to(device, non_blocking=True)
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(videos)
            all_logits.append(logits.detach().float().cpu())
            all_labels.append(labels.detach().cpu())
    if calibrator is not None and hasattr(core, "calibrator"):
        core.calibrator = calibrator
    return torch.cat(all_logits), torch.cat(all_labels)


def calibrate_temperature_bias(model, logits, labels, device, max_iter=50):
    """Optimize (T, b) on cached logits using LBFGS."""
    model.eval()
    core = model.core if hasattr(model, "core") else model
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(core, "calibrator"):
        for p in core.calibrator.parameters():
            p.requires_grad = True

    logits = logits.to(device)
    labels = labels.to(device)

    optimizer = torch.optim.LBFGS(core.calibrator.parameters(), lr=0.5, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        calibrated = core.calibrator(logits)
        loss = F.cross_entropy(calibrated, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    for p in model.parameters():
        p.requires_grad = True


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
    scheduler,
    device,
    eval_dataloader,
    eval_every,
    log_every,
    checkpoint_dir,
    logger,
    rank,
    world_size,
    model_type,
    videomae_model_id,
    base_global_samples,
    next_eval,
    next_milestone,
    ensemble_eval_model=None,
):
    model.train()
    total_loss = 0
    total_samples = 0
    metrics = Metrics()

    pbar = tqdm(dataloader, desc="Training") if rank == 0 else dataloader
    for batch_idx, (videos, labels, dataset_names) in enumerate(pbar):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(videos)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        total_samples += len(labels)

        with torch.no_grad():
            probabilities = F.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)
        metrics.update(predictions, labels, probabilities, dataset_names)

        if rank == 0 and hasattr(pbar, "set_postfix"):
            if batch_idx % log_every == 0:
                train_metrics = metrics.compute()
                pbar.set_postfix(
                    {
                        "loss": f"{total_loss/(batch_idx+1):.4f}",
                        "sn34": f"{train_metrics['sn34']:.4f}",
                        "gap": f"{train_metrics['balance_gap']:.1f}%",
                    }
                )
            else:
                pbar.set_postfix({"loss": f"{total_loss/(batch_idx+1):.4f}"})

        global_samples = base_global_samples + (total_samples * world_size)
        do_eval = global_samples >= next_eval
        if world_size > 1 and dist.is_initialized():
            batch_global = len(labels) * world_size
            near = int((global_samples + batch_global) >= next_eval)
            near_t = torch.tensor(near, device=device)
            dist.all_reduce(near_t, op=dist.ReduceOp.MAX)
            if bool(near_t.item()):
                do_eval_t = torch.tensor(int(do_eval), device=device)
                dist.all_reduce(do_eval_t, op=dist.ReduceOp.MIN)
                do_eval = bool(do_eval_t.item())
            else:
                do_eval = False
        if do_eval and world_size > 1:
            dist.barrier()
        if do_eval and rank == 0:
            sync_samples = global_samples
            if world_size > 1 and dist.is_initialized():
                sync_t = torch.tensor(sync_samples, device=device, dtype=torch.long)
                dist.all_reduce(sync_t, op=dist.ReduceOp.MIN)
                sync_samples = int(sync_t.item())
            logger.info(f"\n{'='*70}")
            logger.info(f"Evaluation at {sync_samples:,} global samples")
            logger.info(f"{'='*70}")

            val_metrics = evaluate(
                model.module if world_size > 1 else model,
                eval_dataloader,
                device,
                ensemble_model=ensemble_eval_model,
            )
            train_metrics = metrics.compute()

            logger.info("Training Metrics:")
            logger.info(f"  SN34:     {train_metrics['sn34']:.4f}")
            logger.info(f"  Accuracy: {train_metrics['accuracy']:.2f}%")
            logger.info(f"  Real:     {train_metrics['real_accuracy']:.2f}%")
            logger.info(f"  Fake:     {train_metrics['fake_accuracy']:.2f}%")
            logger.info(f"  Gap:      {train_metrics['balance_gap']:.2f}%")
            logger.info(f"  MCC:      {train_metrics['mcc']:.4f}")
            logger.info(f"  BCE:      {train_metrics['bce']:.4f}")
            logger.info(f"  FN Rate:  {train_metrics['fn_rate']:.2f}%")

            logger.info("\nValidation Metrics:")
            logger.info(f"  SN34:     {val_metrics['sn34']:.4f} ⭐")
            logger.info(f"  Accuracy: {val_metrics['accuracy']:.2f}%")
            logger.info(f"  Real:     {val_metrics['real_accuracy']:.2f}%")
            logger.info(f"  Fake:     {val_metrics['fake_accuracy']:.2f}%")
            logger.info(f"  Gap:      {val_metrics['balance_gap']:.2f}%")
            logger.info(f"  MCC:      {val_metrics['mcc']:.4f}")
            logger.info(f"  BCE:      {val_metrics['bce']:.4f}")
            logger.info(f"  MCC Score: {val_metrics['mcc_score']:.4f}")
            logger.info(f"  CE Score:  {val_metrics['ce_score']:.4f}")
            logger.info(f"  FN Rate:  {val_metrics['fn_rate']:.2f}%")

            model_state = model.module.state_dict() if world_size > 1 else model.state_dict()
            checkpoint_data = {
                "samples": sync_samples,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": val_metrics,
                "model_type": model_type,
                "videomae_model_id": videomae_model_id,
            }

            checkpoint_path = checkpoint_dir / "best_sn34.pt"
            if checkpoint_path.exists():
                old = torch.load(checkpoint_path, map_location="cpu")
                old_sn34 = old.get("metrics", {}).get("sn34", -1.0)
            else:
                old_sn34 = -1.0
            if val_metrics["sn34"] > old_sn34:
                torch.save(checkpoint_data, checkpoint_path)
                logger.info(
                    f"✅ Saved best_sn34.pt (SN34: {val_metrics['sn34']:.4f}, prev {old_sn34:.4f})"
                )

            checkpoint_path = checkpoint_dir / "best_balanced.pt"
            if checkpoint_path.exists():
                old = torch.load(checkpoint_path, map_location="cpu")
                old_gap = old.get("metrics", {}).get("balance_gap", float("inf"))
            else:
                old_gap = float("inf")
            if val_metrics["balance_gap"] < old_gap:
                torch.save(checkpoint_data, checkpoint_path)
                logger.info(f"✅ Saved best_balanced.pt (Gap: {val_metrics['balance_gap']:.2f}%)")

            if sync_samples >= next_milestone:
                milestone_path = checkpoint_dir / f"checkpoint_{sync_samples//1000}k.pt"
                torch.save(checkpoint_data, milestone_path)
                logger.info(f"💾 Saved milestone: {milestone_path.name}")

            logger.info(f"{'='*70}\n")
            model.train()
        if do_eval:
            next_eval += eval_every
            next_milestone += 500000
        if do_eval and world_size > 1:
            dist.barrier()

    if world_size > 1 and dist.is_initialized():
        global_samples_t = torch.tensor(global_samples, device=device, dtype=torch.long)
        dist.all_reduce(global_samples_t, op=dist.ReduceOp.MIN)
        global_samples = int(global_samples_t.item())
    return metrics.compute(), global_samples, next_eval, next_milestone


# ===========================================
# MAIN
# ===========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Training CSV (shuffled)")
    parser.add_argument("--val-csv", type=str, required=True, help="Validation CSV (hard datasets)")
    parser.add_argument(
        "--model",
        type=str,
        default="videomae_v2",
        choices=["videomae_v2", "mvit_v2", "swin_l", "swin3d_l"],
    )
    parser.add_argument("--videomae-model-id", type=str, default="MCG-NJU/videomae-v2-large")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--eval-every", type=int, default=100000)
    parser.add_argument("--log-every", type=int, default=200, help="Batches between metric summaries")
    parser.add_argument("--output-dir", type=str, default="checkpoints/videomae_v2")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--focal-loss", action="store_true")
    parser.add_argument("--focal-alpha", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--init-temperature", type=float, default=1.0)
    parser.add_argument("--init-bias", type=float, default=0.0)
    parser.add_argument("--calibrate-posthoc", action="store_true", help="Optimize (T,b) on val set")
    parser.add_argument("--ensemble-mvit-checkpoint", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    if is_main:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(output_dir / "training.log"), logging.StreamHandler(sys.stdout)],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.ERROR)

    device = torch.device(f"cuda:{local_rank}")
    if is_main:
        logger.info(f"Model: {args.model}")
        logger.info(f"Device: cuda:{local_rank} (rank {rank}/{world_size})")

    data_manager = BackblazeDataManager(rank=rank)

    if is_main:
        logger.info(f"\nReading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    total_samples = len(df)
    if is_main:
        logger.info(f"Total samples: {total_samples:,}")
        logger.info(f"Real: {(df['label'] == 0).sum():,}")
        logger.info(f"Fake: {(df['label'] == 1).sum():,}")

    model = build_model(args).to(device)
    # Freeze calibrator during main training (post-hoc calibration only)
    core = model.core if hasattr(model, "core") else model
    if hasattr(core, "calibrator"):
        for p in core.calibrator.parameters():
            p.requires_grad = False
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    if args.focal_loss:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(total_samples / (args.batch_size * world_size))
    total_steps = steps_per_epoch * args.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.5,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=100.0,
    )
    scaler = GradScaler()

    num_steps = (total_samples + SAMPLES_PER_STEP - 1) // SAMPLES_PER_STEP

    if is_main:
        logger.info(f"Total steps: {num_steps}")

    if is_main:
        logger.info("\nDownloading validation data...")
    val_df = pd.read_csv(args.val_csv)
    val_file_list = [row["bgr_path"] for _, row in val_df.iterrows()]
    data_manager.download_step_files(val_file_list, -1)

    if world_size > 1:
        dist.barrier()

    val_dataset = StreamingVideoDataset(
        args.val_csv,
        data_manager,
        df=val_df,
        verbose=is_main,
        training=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    ensemble_eval_model = None
    if args.ensemble_mvit_checkpoint and is_main:
        logger.info(f"Loading ensemble MViTv2 checkpoint: {args.ensemble_mvit_checkpoint}")
        ensemble_eval_model = load_checkpoint_model("mvit_v2", args.ensemble_mvit_checkpoint, device)

    global_samples_processed = 0
    next_eval = args.eval_every
    next_milestone = 500000

    for epoch in range(args.epochs):
        if is_main:
            logger.info(f"\n{'='*70}")
            logger.info(f"EPOCH {epoch+1}/{args.epochs}")
            logger.info(f"{'='*70}")

        for step in range(num_steps):
            global_step = epoch * num_steps + step
            if is_main:
                logger.info(f"\n{'='*70}")
                logger.info(f"STEP {step+1}/{num_steps}")
                logger.info(f"{'='*70}")

            start_idx = step * SAMPLES_PER_STEP
            end_idx = min((step + 1) * SAMPLES_PER_STEP, total_samples)
            step_samples = end_idx - start_idx

            step_file_list = [df.iloc[i]["bgr_path"] for i in range(start_idx, end_idx)]
            data_manager.download_step_files(step_file_list, global_step)

            if step < num_steps - 1:
                next_start_idx = (step + 1) * SAMPLES_PER_STEP
                next_end_idx = min((step + 2) * SAMPLES_PER_STEP, total_samples)
                next_file_list = [df.iloc[i]["bgr_path"] for i in range(next_start_idx, next_end_idx)]
                data_manager.prefetch_step_background(next_file_list, global_step + 1)

            if world_size > 1:
                dist.barrier()

            if is_main:
                logger.info(f"Step {step+1}: Training on {step_samples:,} samples")

            df_slice = df.iloc[start_idx:end_idx]
            train_dataset = StreamingVideoDataset(
                args.csv,
                data_manager,
                df=df_slice,
                step=step,
                verbose=is_main,
                training=True,
            )

            if world_size > 1:
                train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            else:
                train_sampler = None
            if train_sampler is not None:
                train_sampler.set_epoch(global_step)

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                shuffle=(train_sampler is None),
                num_workers=8,
                pin_memory=True,
                prefetch_factor=2,
            )

            train_metrics, global_samples_processed, next_eval, next_milestone = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                scheduler,
                device,
                val_loader,
                args.eval_every,
                args.log_every,
                output_dir,
                logger,
                rank,
                world_size,
                args.model,
                args.videomae_model_id,
                global_samples_processed,
                next_eval,
                next_milestone,
                ensemble_eval_model=ensemble_eval_model,
            )

            if is_main:
                logger.info(f"\nStep {step+1} Complete:")
                logger.info(f"  SN34:     {train_metrics['sn34']:.4f}")
                logger.info(f"  Accuracy: {train_metrics['accuracy']:.2f}%")
                logger.info(f"  Real:     {train_metrics['real_accuracy']:.2f}%")
                logger.info(f"  Fake:     {train_metrics['fake_accuracy']:.2f}%")
                logger.info(f"  Gap:      {train_metrics['balance_gap']:.2f}%")

            data_manager.cleanup_old_steps(global_step)

            if world_size > 1:
                dist.barrier()

    if is_main:
        logger.info("\nSaving final checkpoint...")
        model_state = model.module.state_dict() if world_size > 1 else model.state_dict()
        global_samples = global_samples_processed
        final_checkpoint = {
            "samples": global_samples,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": train_metrics,
            "model_type": args.model,
            "videomae_model_id": args.videomae_model_id,
        }
        final_path = output_dir / "last.pt"
        torch.save(final_checkpoint, final_path)
        logger.info("✅ Saved last.pt")

    data_manager.wait_for_prefetch()

    if args.calibrate_posthoc and is_main:
        logger.info("Running post-hoc (T,b) calibration on validation set...")
        calib_model = build_model(args).to(device)
        checkpoint = torch.load(output_dir / "best_sn34.pt", map_location=device)
        calib_model.load_state_dict(checkpoint["model_state_dict"])
        calib_core = calib_model.core if hasattr(calib_model, "core") else calib_model
        if hasattr(calib_core, "calibrator"):
            calib_core.calibrator.log_temperature.data.fill_(0.0)
            calib_core.calibrator.fake_bias.data.fill_(0.0)
        logits, labels = collect_logits_and_labels(calib_model, val_loader, device)
        calibrate_temperature_bias(calib_model, logits, labels, device)
        checkpoint["model_state_dict"] = calib_model.state_dict()
        checkpoint["videomae_model_id"] = args.videomae_model_id
        torch.save(checkpoint, output_dir / "best_sn34.pt")
        logger.info("✅ Updated best_sn34.pt with calibrated (T,b)")

    if world_size > 1:
        dist.barrier()

    if is_main:
        logger.info("\n✅ TRAINING COMPLETE")

    cleanup_ddp()


if __name__ == "__main__":
    main()
