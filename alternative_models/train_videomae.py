#!/usr/bin/env python3
"""
VideoMAE v2 ViT-L training for SN34 (no flash_attn dependency).
Includes: Brier scoring, SWA, streaming, temperature sweep.
"""

import argparse
import copy
import datetime
import math
import os
import psutil
import random
import subprocess
import threading
import time
import warnings
from pathlib import Path

# Set NCCL timeout for long-running operations (streaming downloads can take 30+ minutes)
os.environ.setdefault('NCCL_TIMEOUT', '7200')  # 2 hour timeout
os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')  # Back-compat for older PyTorch
os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')  # Back-compat for older PyTorch
os.environ.setdefault('TORCH_NCCL_BLOCKING_WAIT', '1')  # Preferred env for newer PyTorch
os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')  # Preferred env for newer PyTorch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim import AdamW
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm

# Suppress transformers warnings during model loading
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*ignore_mismatched_sizes.*")

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, AutoModel, AutoConfig
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

B2_BUCKET = "b2:sn33-bucket"
LOCAL_CACHE = os.path.expanduser("~/bittensor/epoch_cache")
SAMPLES_PER_STEP = 50000
MAX_CACHE_STEPS = 1


# ---------------------------------------------------------------------------
# Focal loss & robustness augmentations
# Research basis: "Generalized Design Choices for Deepfake Detectors"
# (Pellegrini et al., Nov 2025) + DeMamba (GenVideo) + DFFC curriculum
# ---------------------------------------------------------------------------

def focal_cross_entropy(logits: torch.Tensor, labels: torch.Tensor,
                        gamma: float = 2.0, label_smoothing: float = 0.0) -> torch.Tensor:
    """Focal loss variant of cross-entropy (Lin et al., 2017).

    Down-weights well-classified (easy) examples by factor (1-p_t)^gamma,
    focusing gradient on hard examples — exactly what's needed to push
    per-dataset accuracy from ~90% to >95%.  At gamma=0 this is standard CE.
    """
    ce = F.cross_entropy(logits, labels, reduction="none",
                         label_smoothing=label_smoothing)
    if gamma == 0.0:
        return ce.mean()
    p_t = torch.exp(-ce)  # p_t = probability of correct class
    focal_weight = (1.0 - p_t) ** gamma
    return (focal_weight * ce).mean()


import io
from PIL import Image


def jpeg_compress_frame(frame: torch.Tensor, quality: int) -> torch.Tensor:
    """JPEG compress+decompress a single frame.

    frame: [C, H, W] float [0, 1]
    Returns: [C, H, W] float [0, 1] with JPEG artifacts.

    This is THE single most impactful augmentation for deepfake detection
    generalization (+4.4% AUROC average, Pellegrini et al. 2025).
    Real-world videos undergo multiple JPEG/codec re-compressions during
    distribution; training with JPEG augmentation teaches the model to
    distinguish generator artifacts from compression artifacts.
    """
    pil_img = TF.to_pil_image(frame.clamp(0, 1).cpu())
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    pil_img = Image.open(buf)
    return TF.to_tensor(pil_img).to(frame.device)


def apply_jpeg_augmentation(videos: torch.Tensor, prob: float,
                            quality_min: int, quality_max: int) -> torch.Tensor:
    """Apply JPEG compression to random videos in a batch.

    videos: [B, T, C, H, W] float [0, 1]
    Same quality applied to ALL frames of a video (temporal consistency).
    """
    B, T = videos.shape[:2]
    for i in range(B):
        if random.random() < prob:
            q = random.randint(quality_min, quality_max)
            for t in range(T):
                videos[i, t] = jpeg_compress_frame(videos[i, t], q)
    return videos


def apply_gaussian_noise(videos: torch.Tensor, prob: float,
                         std_range: tuple = (0.005, 0.03)) -> torch.Tensor:
    """Add Gaussian noise to random videos. Pure tensor op, fast on GPU."""
    B = videos.shape[0]
    for i in range(B):
        if random.random() < prob:
            std = random.uniform(*std_range)
            videos[i] = (videos[i] + torch.randn_like(videos[i]) * std).clamp(0, 1)
    return videos


def apply_gaussian_blur(videos: torch.Tensor, prob: float,
                        kernel_sizes: tuple = (3, 5)) -> torch.Tensor:
    """Apply Gaussian blur to random videos. Consistent across frames."""
    B, T = videos.shape[:2]
    for i in range(B):
        if random.random() < prob:
            k = random.choice(kernel_sizes)
            for t in range(T):
                videos[i, t] = TF.gaussian_blur(videos[i, t], kernel_size=k)
    return videos


def setup_ddp():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # Bind this worker to its CUDA device before NCCL group init.
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(hours=2),
        )
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


class VideoMAEv2ForClassification(nn.Module):
    """
    Wrapper for VideoMAEv2-giant (AutoModel) with classification head.
    VideoMAEv2-giant is a feature extraction model, we add classifier on top.
    """
    def __init__(self, model_id: str, num_labels: int = 2):
        super().__init__()
        print(f"[VideoMAEv2ForClassification] Loading config from {model_id}")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(f"[VideoMAEv2ForClassification] Config type: {type(config).__name__}")
        
        # Debug: Print all config attributes
        if hasattr(config, 'hidden_size'):
            print(f"[VideoMAEv2ForClassification] config.hidden_size = {config.hidden_size}")
        if hasattr(config, 'd_model'):
            print(f"[VideoMAEv2ForClassification] config.d_model = {config.d_model}")
        if hasattr(config, 'encoder_width'):
            print(f"[VideoMAEv2ForClassification] config.encoder_width = {config.encoder_width}")
        
        print(f"[VideoMAEv2ForClassification] Loading model from {model_id}")
        # Fix for meta tensor bug in VideoMAEv2's custom code with PyTorch 2.10
        # Disable accelerate's meta device usage
        import accelerate
        original_env = os.environ.get('ACCELERATE_USE_CPU')
        os.environ['ACCELERATE_USE_CPU'] = '1'
        
        try:
            self.videomae = AutoModel.from_pretrained(
                model_id, 
                config=config, 
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
                device_map=None,
            )
        finally:
            # Restore original env
            if original_env is None:
                os.environ.pop('ACCELERATE_USE_CPU', None)
            else:
                os.environ['ACCELERATE_USE_CPU'] = original_env
        
        # Get hidden size from config or infer from model
        if hasattr(config, 'hidden_size'):
            hidden_size = config.hidden_size
        elif hasattr(config, 'd_model'):
            hidden_size = config.d_model
        elif hasattr(config, 'encoder_width'):
            hidden_size = config.encoder_width
        else:
            # Detect from model parameters
            hidden_size = None
            for name, param in self.videomae.named_parameters():
                if 'blocks.0.norm1.weight' in name:
                    hidden_size = param.shape[0]
                    print(f"[VideoMAEv2ForClassification] Detected from blocks.0.norm1: {hidden_size}")
                    break
                elif 'encoder.layer.0' in name and 'weight' in name:
                    hidden_size = param.shape[-1]
                    print(f"[VideoMAEv2ForClassification] Detected from encoder.layer.0: {hidden_size}")
                    break
            if hidden_size is None:
                hidden_size = 1408  # VideoMAEv2-giant default
                print(f"[VideoMAEv2ForClassification] Using default: {hidden_size}")
        
        print(f"[VideoMAEv2ForClassification] Final hidden_size: {hidden_size}")
        
        # Add classification head (fc_norm + linear)
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Initialize classifier
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, pixel_values, **kwargs):
        outputs = self.videomae(pixel_values, **kwargs)
        # VideoMAE-v2 uses mean pooling
        if hasattr(outputs, 'last_hidden_state'):
            hidden_state = outputs.last_hidden_state
            sequence_output = hidden_state.mean(dim=1)
        else:
            sequence_output = outputs
        
        if self.fc_norm is not None:
            sequence_output = self.fc_norm(sequence_output)
        logits = self.classifier(sequence_output)
        
        # Return object with .logits attribute
        class Output:
            def __init__(self, logits):
                self.logits = logits
        return Output(logits)


class VideoNpyDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        data_root: str = None,
        data_manager=None,
        mmap: bool = False,
        temporal_skip_prob: float = 0.0,
    ):
        self.df = pd.read_csv(csv_path)
        self.data_root = Path(data_root) if data_root else None
        self.data_manager = data_manager
        self.mmap = mmap
        self.temporal_skip_prob = temporal_skip_prob
        self.return_dataset_idx = False
        self._dataset_to_idx = {}

    def enable_dataset_tracking(self):
        """Enable returning dataset index as third element from __getitem__.
        Call this before creating a DataLoader for macro-averaged evaluation."""
        if "dataset" in self.df.columns:
            unique_ds = sorted(self.df["dataset"].unique())
            self._dataset_to_idx = {ds: i for i, ds in enumerate(unique_ds)}
            self.return_dataset_idx = True
        return self._dataset_to_idx

    def disable_dataset_tracking(self):
        self.return_dataset_idx = False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        max_retries = 3
        original_label = None
        dataset_idx = 0

        for attempt in range(max_retries):
            try:
                row = self.df.iloc[idx]
                bgr_path = row["bgr_path"]
                label = int(row["label"])

                if original_label is None:
                    original_label = label
                    if self.return_dataset_idx and "dataset" in row.index:
                        dataset_idx = self._dataset_to_idx.get(row["dataset"], 0)

                if self.data_manager is not None:
                    bgr_path = self.data_manager.local_cache / bgr_path
                elif self.data_root:
                    bgr_path = self.data_root / bgr_path

                mmap_mode = "r" if self.mmap else None
                video_bgr = np.load(bgr_path, mmap_mode=mmap_mode)

                if video_bgr.shape != (16, 224, 224, 3):
                    raise ValueError(f"Wrong shape: {video_bgr.shape}")

                video = torch.from_numpy(video_bgr)
                video = video.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
                video = video[:, [2, 1, 0], :, :]  # BGR -> RGB

                if self.temporal_skip_prob > 0 and torch.rand(1).item() < self.temporal_skip_prob:
                    video = video[::2]
                    if video.shape[0] < 16:
                        last = video[-1]
                        pad = 16 - video.shape[0]
                        video = torch.cat([video, last.unsqueeze(0).repeat(pad, 1, 1, 1)], dim=0)
                    else:
                        video = video[:16]

                if self.return_dataset_idx:
                    return video, label, dataset_idx
                return video, label
            except Exception as e:
                if attempt < max_retries - 1:
                    idx = (idx + hash(str(e)) % len(self.df)) % len(self.df)
                else:
                    print(f"[Dataset] ERROR after {max_retries} retries for idx {idx}: {e}")
                    print(f"[Dataset] Returning dummy black video with correct label={original_label}")
                    dummy = torch.zeros(16, 3, 224, 224, dtype=torch.uint8)
                    lbl = original_label if original_label is not None else 0
                    if self.return_dataset_idx:
                        return dummy, lbl, dataset_idx
                    return dummy, lbl


def apply_color_jitter(video: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Apply color jitter to a single video. 
    video: [T, C, H, W] - single video tensor
    Returns: [T, C, H, W] with same jitter applied to all frames (temporal consistency)
    """
    if strength <= 0:
        return video
    brightness = float(torch.empty(1).uniform_(max(0.0, 1.0 - strength), 1.0 + strength))
    contrast = float(torch.empty(1).uniform_(max(0.0, 1.0 - strength), 1.0 + strength))
    saturation = float(torch.empty(1).uniform_(max(0.0, 1.0 - strength), 1.0 + strength))
    frames = []
    for f in video:
        f = TF.adjust_brightness(f, brightness)
        f = TF.adjust_contrast(f, contrast)
        f = TF.adjust_saturation(f, saturation)
        frames.append(f)
    return torch.stack(frames, dim=0)


def apply_horizontal_flip(video: torch.Tensor) -> torch.Tensor:
    return video.flip(dims=[-1])


def extract_logits(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def load_teacher(args, device, is_main: bool):
    if not args.teacher_model_id:
        return None
    if is_main:
        print(f"[Teacher] Loading {args.teacher_type} from {args.teacher_model_id}")
    if args.teacher_type == "videomae":
        teacher = VideoMAEForVideoClassification.from_pretrained(
            args.teacher_model_id,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
    elif args.teacher_type == "internvideo2":
        teacher = torch.hub.load(
            args.teacher_repo,
            args.teacher_entrypoint,
            pretrained=True,
            trust_repo=True,
        )
    elif args.teacher_type == "internvideo2_hf":
        teacher = AutoModel.from_pretrained(
            args.teacher_model_id,
            trust_remote_code=True,
        )
    elif args.teacher_type == "internvideo2_local":
        import sys
        code_path = os.path.expanduser(args.internvideo2_code_path)
        if not os.path.isdir(code_path):
            raise RuntimeError(
                f"InternVideo2 code path not found: {code_path}. "
                "Set --internvideo2-code-path to the repo's single_modality folder "
                "(e.g. ~/InternVideo/InternVideo2/single_modality) or change --teacher-type."
            )
        sys.path.insert(0, code_path)
        from models.internvideo2_teacher import InternVideo2
        
        # Create InternVideo2-6B model directly (configured for 8 frames as per checkpoint)
        teacher = InternVideo2(
            img_size=224, patch_size=14, embed_dim=3200,
            depth=48, num_heads=25, mlp_ratio=4,
            attn_pool_num_heads=16, clip_embed_dim=768,
            clip_norm_type='l2',
            return_attn=True,
            clip_return_layer=1,
            clip_return_interval=1,
            num_frames=8,  # Checkpoint is for 8 frames
            tubelet_size=1,
        )
        
        # Load the downloaded stage2 weights
        checkpoint = torch.load(args.teacher_model_id, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        elif 'module' in checkpoint:
            checkpoint = checkpoint['module']
        message = teacher.load_state_dict(checkpoint, strict=False)
        if is_main:
            print(f"[Teacher] Loaded weights: {message}")
    else:
        raise ValueError(f"Unknown teacher_type: {args.teacher_type}")
    teacher = teacher.to(device)
    teacher.eval()
    # Freeze all teacher parameters and disable gradient computation
    for p in teacher.parameters():
        p.requires_grad = False
    # Ensure teacher stays in eval mode
    teacher.requires_grad_(False)
    return teacher


def sample_frames_for_teacher(videos: torch.Tensor, target_frames: int) -> torch.Tensor:
    """
    Sample target_frames from videos for teacher that expects fewer frames.
    videos: [B, T, C, H, W]
    Returns: [B, target_frames, C, H, W]
    """
    T = videos.shape[1]
    if target_frames <= 0 or target_frames >= T:
        return videos
    # Uniform sampling
    indices = torch.linspace(0, T - 1, target_frames).long()
    return videos[:, indices]


def cosine_similarity_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute 1 - cosine_similarity as a loss (0 = identical direction, 2 = opposite).
    a, b: [B, D] feature tensors
    """
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)
    cosine_sim = (a_norm * b_norm).sum(dim=-1)  # [B]
    return (1 - cosine_sim).mean()


def get_teacher_features(teacher, videos: torch.Tensor, teacher_type: str = "") -> torch.Tensor:
    """
    Extract features from teacher model.
    For InternVideo2-6B HF: use get_vid_feat() which returns [B, D] feature.
    Input: [B, T, C, H, W] format
    """
    if hasattr(teacher, "get_vid_feat"):
        # InternVideo2 HF model - expects [B, T, C, H, W] in [0, 1] range
        return teacher.get_vid_feat(videos)
    
    # Local InternVideo2 models expect [B, C, T, H, W]
    if "internvideo2" in teacher_type.lower():
        videos = videos.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
    
    out = teacher(videos)
    
    # InternVideo2 returns tuple: (intermediate_features, final_feature, attention) or (intermediate_features, final_feature)
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        # out[0] = intermediate features [K, B, THW+1, C]
        # out[1] = final pooled feature [B, dim] <- This is what we want
        return out[1]
    
    if hasattr(out, "last_hidden_state"):
        hidden = out.last_hidden_state
        return hidden[:, 0] if hidden.dim() == 3 and hidden.size(1) >= 1 else hidden.mean(dim=1)
    if hasattr(out, "video_embeds"):
        return out.video_embeds
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def stratified_shuffle_df(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Shuffle DataFrame so that each contiguous block of N rows contains
    proportional representation from every dataset.  This ensures that when
    the training loop slices `step_df = df[start:end]`, each step sees ALL
    datasets rather than being dominated by large ones.

    Algorithm – interleaved round-robin with jitter:
    1. Group rows by 'dataset'.
    2. Shuffle each group independently.
    3. Interleave: walk through groups in proportion, picking one row at a
       time from the next dataset in round-robin order.  Smaller datasets
       wrap around (repeat) to fill their proportional slots.
    """
    if "dataset" not in df.columns or len(df) == 0:
        return df.sample(frac=1, random_state=seed).reset_index(drop=True) if len(df) > 0 else df.reset_index(drop=True)

    rng = np.random.RandomState(seed)
    # Fill NaN dataset values to avoid sort/comparison errors
    if df["dataset"].isna().any():
        df = df.copy()
        df["dataset"] = df["dataset"].fillna("_unknown_")
    datasets = sorted(df["dataset"].unique())
    num_datasets = len(datasets)
    if num_datasets == 0:
        return df.reset_index(drop=True)
    total = len(df)

    # Shuffle each group and store as list of row-indices
    group_indices = {}
    for ds in datasets:
        idx = df.index[df["dataset"] == ds].tolist()
        rng.shuffle(idx)
        group_indices[ds] = idx

    # Compute per-dataset weight (= share of total).
    # Give each dataset a minimum floor to ensure even tiny datasets appear
    # in every step block.
    min_weight = 1.0 / (num_datasets * 2)  # floor = 0.5/N of block
    raw_weights = {ds: max(len(group_indices[ds]) / total, min_weight) for ds in datasets}
    weight_sum = sum(raw_weights.values())
    norm_weights = {ds: w / weight_sum for ds, w in raw_weights.items()}

    # Build interleaved index list
    interleaved = []
    cursors = {ds: 0 for ds in datasets}
    # Number of rows each dataset should contribute
    contributions = {ds: max(1, int(round(norm_weights[ds] * total))) for ds in datasets}
    # Adjust to match total exactly
    diff = total - sum(contributions.values())
    sorted_ds = sorted(datasets, key=lambda d: len(group_indices[d]), reverse=True)
    for i in range(abs(diff)):
        ds = sorted_ds[i % len(sorted_ds)]
        contributions[ds] += 1 if diff > 0 else -1

    # Round-robin: distribute each dataset's contribution evenly across the stream
    # Use a priority queue approach: dataset with most remaining gets picked next
    import heapq
    heap = []
    for ds in datasets:
        if contributions[ds] > 0:
            # Priority = position where next sample should go (evenly spaced)
            spacing = total / contributions[ds]
            heapq.heappush(heap, (spacing / 2, ds, spacing))

    while heap and len(interleaved) < total:
        pos, ds, spacing = heapq.heappop(heap)
        idx_list = group_indices[ds]
        cursor = cursors[ds]
        # Wrap around for small datasets
        actual_idx = idx_list[cursor % len(idx_list)]
        interleaved.append(actual_idx)
        cursors[ds] = cursor + 1
        if cursors[ds] < contributions[ds]:
            heapq.heappush(heap, (pos + spacing, ds, spacing))

    # Fill any remaining (shouldn't happen, but safety)
    while len(interleaved) < total:
        interleaved.append(df.index[len(interleaved) % total])

    return df.iloc[interleaved[:total]].reset_index(drop=True)


def compute_dataset_weights(df: pd.DataFrame, max_repeat: float = 20.0) -> np.ndarray:
    """Compute per-sample weights so that each dataset contributes more
    equally to the expected gradient, regardless of dataset size.

    The `max_repeat` parameter caps how many times a single sample can be
    effectively repeated per epoch via the weighted sampler.  This prevents
    memorization of tiny datasets while still giving them a meaningful boost.

    max_repeat=20 with 3 epochs → each sample seen at most ~60 times.
    With ~35 effective augmented views per sample (hflip × color_jitter ×
    JPEG_quality × gaussian_noise × gaussian_blur × temporal_skip),
    each distinct view is seen ~1.7 times — sufficient for learning the
    generator's artifact signature without memorizing video content.

    Previously max_repeat=50 was used with fewer augmentations (~12 views),
    resulting in ~12.5 repeats per view — enough to memorize content.
    Reducing to 20 with richer augmentation keeps the per-view exposure
    low while still giving small datasets meaningful gradient contribution.

    Returns a numpy array of length len(df) with per-sample weights.
    Within each dataset, real and synthetic are also balanced.
    """
    if "dataset" not in df.columns or len(df) == 0:
        return np.ones(max(len(df), 1), dtype=np.float64)[:len(df)] if len(df) > 0 else np.array([], dtype=np.float64)

    weights = np.ones(len(df), dtype=np.float64)
    # Handle NaN dataset values
    if df["dataset"].isna().any():
        df = df.copy()
        df["dataset"] = df["dataset"].fillna("_unknown_")
    datasets = df["dataset"].unique()
    num_datasets = len(datasets)
    if num_datasets == 0:
        return weights

    for ds in datasets:
        ds_mask = (df["dataset"] == ds).values
        ds_count = ds_mask.sum()
        if ds_count == 0:
            continue
        ds_weight = (1.0 / num_datasets) / ds_count

        ds_indices = np.where(ds_mask)[0]
        labels = df.iloc[ds_indices]["label"].values
        n_real = (labels == 0).sum()
        n_synth = (labels == 1).sum()

        if n_real > 0 and n_synth > 0:
            half_budget = (1.0 / num_datasets) / 2.0
            for idx in ds_indices:
                lbl = df.iloc[idx]["label"]
                if lbl == 0:
                    weights[idx] = half_budget / n_real
                else:
                    weights[idx] = half_budget / n_synth
        else:
            weights[ds_indices] = ds_weight

    # Normalize so mean weight = 1.0 (preserves effective batch size)
    weights /= weights.mean()

    # Cap maximum weight to prevent memorization of tiny datasets.
    # max_repeat=20 means a sample can be sampled at most ~20x per epoch.
    # Without cap, a 91-sample dataset in a 903K CSV would get weight ~191x,
    # causing each sample to be seen ~573 times across 3 epochs.
    if max_repeat > 0:
        weights = np.clip(weights, 0.0, max_repeat)
        weights /= weights.mean()  # Re-normalize after clip

    return weights


def _compute_sn34(mcc, brier, alpha=1.2, beta=1.8):
    """Compute SN34 score from MCC and Brier."""
    mcc_norm = max(0.0, min((mcc + 1.0) / 2.0, 1.0)) ** alpha
    brier_norm = max(0.0, (0.25 - brier) / 0.25) ** beta
    return math.sqrt(max(1e-12, mcc_norm * brier_norm))


def _compute_mcc(tp, tn, fp, fn):
    """Compute Matthews Correlation Coefficient."""
    numerator = (tp * tn) - (fp * fn)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + tn + fp + fn) > 0 else 0
    return numerator / (denom + 1e-8)


# Per-dataset trend tracking across evaluation rounds.
# Detects regression: a dataset that was performing well starts degrading.
_dataset_trend_history = {}  # {dataset_name: [(step, sn34), ...]}


def _track_dataset_trends(eval_info: dict, step: int, is_main: bool):
    """Track per-dataset SN34 across evaluations and warn on regression.

    Key insight from Community Forensics (CVPR 2025): evaluating per-source
    metrics separately is essential to catch problems. But just printing
    worst-10 is not enough — we need to track TRENDS to detect if a dataset
    that was learned starts being forgotten (catastrophic forgetting at
    dataset level).
    """
    if not is_main or not eval_info.get("per_dataset"):
        return

    regressions = []
    for ds in eval_info["per_dataset"]:
        name = ds["name"]
        sn34 = ds["sn34"]
        if name not in _dataset_trend_history:
            _dataset_trend_history[name] = []
        history = _dataset_trend_history[name]
        history.append((step, sn34))

        if len(history) >= 3:
            prev_best = max(h[1] for h in history[:-1])
            if sn34 < prev_best - 0.10:
                regressions.append((name, prev_best, sn34))

    if regressions:
        print(f"  [REGRESSION WARNING] {len(regressions)} dataset(s) regressing:")
        for name, prev_best, current in regressions[:5]:
            print(f"    {name}: was {prev_best:.4f} → now {current:.4f} "
                  f"(dropped {prev_best - current:.4f})")


def evaluate(
    model,
    dataloader,
    device,
    mean,
    std,
    inference_temp: float = 1.0,
    sn34_alpha: float = 1.2,
    sn34_beta: float = 1.8,
    use_bf16: bool = True,
    world_size: int = 1,
    rank: int = 0,
    is_main: bool = True,
    macro_avg: bool = False,
    dataset_idx_to_name: dict = None,
):
    """Run evaluation. Uses all GPUs when world_size > 1 (distributed eval).

    When macro_avg=True, the dataloader's dataset must have
    return_dataset_idx=True so it yields (video, label, dataset_idx).
    The returned SN34 will be the macro-average across datasets (each
    dataset contributes equally), which prevents large datasets from
    drowning out small ones during checkpoint selection.
    """
    model.eval()
    total = 0
    correct = 0
    all_probs = []
    all_labels = []
    all_ds_ids = []
    tp = tn = fp = fn = 0
    sum_bce = 0.0
    sum_brier = 0.0
    count_bce = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False, disable=(rank != 0)):
            if macro_avg and len(batch) == 3:
                videos, labels, ds_ids = batch
                all_ds_ids.append(ds_ids.detach().cpu())
            else:
                videos, labels = batch[0], batch[1]

            videos = videos.to(device, non_blocking=True).float() / 255.0
            videos = (videos - mean) / std
            labels = labels.to(device, non_blocking=True)

            videos = videos.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(pixel_values=videos)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            logits = logits / max(inference_temp, 1e-6)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.numel()
            all_probs.append(probs[:, 1].detach().cpu())
            all_labels.append(labels.detach().cpu())

            preds_np = preds.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            tp += ((preds_np == 1) & (labels_np == 1)).sum()
            tn += ((preds_np == 0) & (labels_np == 0)).sum()
            fp += ((preds_np == 1) & (labels_np == 0)).sum()
            fn += ((preds_np == 0) & (labels_np == 1)).sum()

    # Aggregate across GPUs when distributed
    if world_size > 1 and dist.is_initialized():
        metrics = torch.tensor([correct, total, tp, tn, fp, fn], dtype=torch.float64, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        correct, total, tp, tn, fp, fn = metrics.tolist()
        correct, total, tp, tn, fp, fn = int(correct), int(total), int(tp), int(tn), int(fp), int(fn)

        probs = torch.cat(all_probs).float().numpy()
        labels_np = torch.cat(all_labels).numpy()
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        sum_bce = np.sum(labels_np * np.log(probs) + (1 - labels_np) * np.log(1 - probs))
        sum_brier = np.sum((probs - labels_np) ** 2)
        count_bce = len(probs)
        local_sums = torch.tensor([float(sum_bce), float(sum_brier), float(count_bce)], dtype=torch.float64, device=device)
        dist.all_reduce(local_sums, op=dist.ReduceOp.SUM)
        sum_bce, sum_brier, count_bce = local_sums.tolist()
        bce = -sum_bce / max(count_bce, 1)
        brier = sum_brier / max(count_bce, 1)

        if macro_avg and all_ds_ids:
            ds_ids_local = torch.cat(all_ds_ids).long().to(device)
            all_ds_gathered = [torch.zeros_like(ds_ids_local) for _ in range(world_size)]
            dist.all_gather(all_ds_gathered, ds_ids_local)
            all_ds_ids_np = torch.cat(all_ds_gathered).cpu().numpy()

            all_probs_local = torch.cat(all_probs).float().to(device)
            all_labels_local = torch.cat(all_labels).long().to(device)
            probs_gathered = [torch.zeros_like(all_probs_local) for _ in range(world_size)]
            labels_gathered = [torch.zeros_like(all_labels_local) for _ in range(world_size)]
            dist.all_gather(probs_gathered, all_probs_local)
            dist.all_gather(labels_gathered, all_labels_local)
            all_probs_np = torch.cat(probs_gathered).cpu().numpy()
            all_labels_np = torch.cat(labels_gathered).cpu().numpy()
    else:
        probs = torch.cat(all_probs).float().numpy()
        labels_np = torch.cat(all_labels).numpy()
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        bce = -np.mean(labels_np * np.log(probs) + (1 - labels_np) * np.log(1 - probs))
        brier = np.mean((probs - labels_np) ** 2)

        if macro_avg and all_ds_ids:
            all_ds_ids_np = torch.cat(all_ds_ids).numpy()
            all_probs_np = probs
            all_labels_np = labels_np

    # Micro-averaged metrics (standard)
    mcc = _compute_mcc(tp, tn, fp, fn)
    micro_sn34 = _compute_sn34(mcc, brier, sn34_alpha, sn34_beta)
    acc = correct / max(total, 1)

    # Confidence distribution diagnostics
    eval_info = {
        "per_dataset": [],
        "min_sn34": 1.0,
        "min_sn34_dataset": "",
        "confidence": {},
    }
    if all_probs:
        all_p = torch.cat(all_probs).float().numpy()
        confidence = np.maximum(all_p, 1.0 - all_p)
        uncertain_mask = (all_p >= 0.4) & (all_p <= 0.6)
        confident_mask = (all_p <= 0.1) | (all_p >= 0.9)
        eval_info["confidence"] = {
            "uncertain_pct": float(uncertain_mask.mean() * 100),
            "confident_pct": float(confident_mask.mean() * 100),
            "mean_confidence": float(confidence.mean()),
        }
        if is_main:
            ci = eval_info["confidence"]
            print(f"  [confidence] uncertain(0.4-0.6)={ci['uncertain_pct']:.1f}% "
                  f"confident(<0.1/>0.9)={ci['confident_pct']:.1f}% "
                  f"mean_conf={ci['mean_confidence']:.3f}")

    # Macro-averaged metrics (each dataset contributes equally)
    if macro_avg and all_ds_ids:
        unique_ds = np.unique(all_ds_ids_np)
        ds_sn34s = []
        ds_mccs = []
        ds_briers = []
        idx_to_name = dataset_idx_to_name or {}
        per_ds_details = []

        for ds_id in sorted(unique_ds):
            mask = all_ds_ids_np == ds_id
            ds_probs = np.clip(all_probs_np[mask], 1e-7, 1 - 1e-7)
            ds_labels = all_labels_np[mask]
            ds_preds = (ds_probs >= 0.5).astype(int)
            ds_n = mask.sum()

            if ds_n < 2:
                continue

            ds_tp = ((ds_preds == 1) & (ds_labels == 1)).sum()
            ds_tn = ((ds_preds == 0) & (ds_labels == 0)).sum()
            ds_fp = ((ds_preds == 1) & (ds_labels == 0)).sum()
            ds_fn = ((ds_preds == 0) & (ds_labels == 1)).sum()
            ds_mcc = _compute_mcc(ds_tp, ds_tn, ds_fp, ds_fn)
            ds_brier = np.mean((ds_probs - ds_labels) ** 2)
            ds_sn34 = _compute_sn34(ds_mcc, ds_brier, sn34_alpha, sn34_beta)

            ds_mccs.append(ds_mcc)
            ds_briers.append(ds_brier)
            ds_sn34s.append(ds_sn34)

            ds_name = idx_to_name.get(int(ds_id), f"ds_{int(ds_id)}")
            ds_acc = float((ds_preds == ds_labels).mean())
            per_ds_details.append({
                "name": ds_name, "n": int(ds_n), "acc": ds_acc,
                "mcc": float(ds_mcc), "brier": float(ds_brier), "sn34": float(ds_sn34),
            })

        if ds_sn34s:
            macro_sn34 = float(np.mean(ds_sn34s))
            macro_mcc = float(np.mean(ds_mccs))
            macro_brier = float(np.mean(ds_briers))

            per_ds_details.sort(key=lambda x: x["sn34"])
            eval_info["per_dataset"] = per_ds_details
            eval_info["min_sn34"] = per_ds_details[0]["sn34"] if per_ds_details else 1.0
            eval_info["min_sn34_dataset"] = per_ds_details[0]["name"] if per_ds_details else ""

            if is_main:
                print(f"  [macro-eval] {len(ds_sn34s)} datasets | "
                      f"macro_sn34={macro_sn34:.4f} macro_mcc={macro_mcc:.4f} "
                      f"macro_brier={macro_brier:.4f} (micro_sn34={micro_sn34:.4f})")
                print(f"  [macro-eval] min_sn34={eval_info['min_sn34']:.4f} "
                      f"({eval_info['min_sn34_dataset']})")
                print(f"  [macro-eval] Worst 10 datasets:")
                for d in per_ds_details[:10]:
                    print(f"    {d['name']:42s}  n={d['n']:5d}  acc={d['acc']:.3f}  "
                          f"mcc={d['mcc']:+.3f}  brier={d['brier']:.4f}  sn34={d['sn34']:.4f}")

            sn34 = macro_sn34
        else:
            sn34 = micro_sn34
    else:
        sn34 = micro_sn34

    return acc, bce, brier, mcc, sn34, eval_info


def get_optimal_rclone_params():
    """Auto-detect optimal rclone parameters based on system resources."""
    cpu_count = os.cpu_count() or 8
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    
    # Scale parameters based on resources
    # More cores = more parallel transfers
    # For many small files: maximize transfers, minimize multi_thread per file
    
    if cpu_count >= 128:
        # Ultra-high CPU count (e.g., 176 CPUs)
        transfers = min(256, cpu_count)  # Very high parallelism for many small files
        checkers = min(256, cpu_count)  # Enable parallel checking for speed
        buffer_size = "256M" if ram_gb >= 512 else "128M"
        multi_thread = 8  # Higher per-file threading for better throughput
    elif ram_gb >= 500 and cpu_count >= 64:
        # High-end server (e.g., 4xH200 with 2TB RAM)
        transfers = min(128, cpu_count)
        checkers = min(128, cpu_count)
        buffer_size = "256M"
        multi_thread = 8
    elif ram_gb >= 128 and cpu_count >= 32:
        # Mid-high server
        transfers = min(96, cpu_count)
        checkers = min(96, cpu_count)
        buffer_size = "128M"
        multi_thread = 6
    elif ram_gb >= 64 and cpu_count >= 16:
        # Mid server
        transfers = min(64, cpu_count)
        checkers = min(64, cpu_count)
        buffer_size = "64M"
        multi_thread = 4
    elif ram_gb >= 32 and cpu_count >= 8:
        # Standard server
        transfers = 32
        checkers = 0
        buffer_size = "32M"
        multi_thread = 4
    else:
        # Minimal/VPS
        transfers = 16
        checkers = 0
        buffer_size = "32M"
        multi_thread = 2
    
    return {
        "transfers": transfers,
        "checkers": checkers,
        "buffer_size": buffer_size,
        "multi_thread": multi_thread,
    }


class BackblazeDataManager:
    def __init__(self, b2_bucket: str = B2_BUCKET, local_cache: str = LOCAL_CACHE, rank: int = 0,
                 rclone_transfers: int = None, rclone_checkers: int = None,
                 rclone_buffer: str = None, rclone_threads: int = None):
        self.b2_bucket = b2_bucket
        self.local_cache = Path(local_cache)
        self.local_cache.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.prefetch_thread = None
        self.prefetch_step = None
        self.step_files = {}
        
        # Auto-detect optimal rclone parameters if not specified
        auto_params = get_optimal_rclone_params()
        self.rclone_transfers = rclone_transfers or auto_params["transfers"]
        self.rclone_checkers = rclone_checkers or auto_params["checkers"]
        self.rclone_buffer = rclone_buffer or auto_params["buffer_size"]
        self.rclone_threads = rclone_threads or auto_params["multi_thread"]
        
        if rank == 0:
            cpu_count = os.cpu_count() or 8
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            print(f"[rclone] System: {cpu_count} CPUs, {ram_gb:.0f}GB RAM")
            print(f"[rclone] Config: transfers={self.rclone_transfers}, checkers={self.rclone_checkers}, "
                  f"buffer={self.rclone_buffer}, threads={self.rclone_threads}")

    def download_files_batch(self, file_paths):
        import tempfile
        import time
        to_download = []
        for f in file_paths:
            local_path = self.local_cache / f
            # Skip only if file exists AND has reasonable size (>100 bytes)
            # This catches incomplete/corrupted downloads
            if not local_path.exists() or local_path.stat().st_size < 100:
                to_download.append(f)
                local_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not to_download:
            print(f"[download] All {len(file_paths)} files already cached")
            return
        
        print(f"[download] Downloading {len(to_download)}/{len(file_paths)} files from Backblaze...")
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
            for f in to_download:
                tf.write(f + "\n")
            files_from_path = tf.name
        
        start_time = time.time()
        try:
            cmd = [
                "rclone", "copy",
                self.b2_bucket,
                str(self.local_cache),
                "--files-from-raw", files_from_path,  # Use raw mode - no listing, just copy the exact files
                "--transfers", str(self.rclone_transfers),
                "--checkers", str(self.rclone_checkers) if self.rclone_checkers > 0 else "0",  # Respect user setting
                "--multi-thread-streams", str(self.rclone_threads),
                "--buffer-size", self.rclone_buffer,
                "--no-traverse",  # Skip directory traversal
                "--size-only",  # Compare file sizes (fast, catches incomplete downloads)
                "--b2-chunk-size", "96M",  # Optimize for B2 upload chunk size
                "--b2-upload-cutoff", "200M",  # Use multipart for files > 200MB
                "--use-mmap",  # Use memory-mapped I/O for better performance
                "--progress",  # Show progress
                "--stats", "5s",  # Update stats every 5 seconds (less overhead than 1s)
                "--stats-one-line",  # Compact display
            ]
            result = subprocess.run(cmd, capture_output=False, timeout=7200)  # 2 hour timeout for large downloads
            elapsed = time.time() - start_time
            if result.returncode == 0:
                print(f"[download] Complete: {len(to_download)} files in {elapsed:.1f}s")
            else:
                print(f"[download] Warning: rclone exit code {result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"[download] Warning: rclone timed out after 7200s, some files may be incomplete")
        except Exception as e:
            print(f"[download] Warning: rclone failed: {e}")
        finally:
            try:
                os.unlink(files_from_path)
            except Exception:
                pass

    def download_step_files(self, file_list, step: int):
        if self.rank != 0:
            return
        print(f"[step {step}] Downloading {len(file_list)} files...")
        self.download_files_batch(file_list)
        
        # Validate downloaded files and re-download if corrupt
        import numpy as np
        corrupted = []
        for f in file_list:
            local_path = self.local_cache / f
            if not local_path.exists():
                corrupted.append(f)
                continue
            try:
                # Quick validation: try to load the array header
                arr = np.load(local_path, mmap_mode='r')
                expected_shape = (16, 224, 224, 3)
                if arr.shape != expected_shape:
                    print(f"[validation] Corrupt file (wrong shape): {f} - shape {arr.shape} != {expected_shape}")
                    corrupted.append(f)
                    local_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"[validation] Corrupt file (load error): {f} - {e}")
                corrupted.append(f)
                local_path.unlink(missing_ok=True)
        
        if corrupted:
            print(f"[validation] Re-downloading {len(corrupted)} corrupted files...")
            self.download_files_batch(corrupted)
            
            # Validate again - if still bad, just warn and skip
            still_bad = []
            for f in corrupted:
                local_path = self.local_cache / f
                try:
                    arr = np.load(local_path, mmap_mode='r')
                    if arr.shape != (16, 224, 224, 3):
                        still_bad.append(f)
                except Exception:
                    still_bad.append(f)
            
            if still_bad:
                print(f"[validation] WARNING: {len(still_bad)} files still corrupt after re-download, will skip during training")
                # Remove from file_list so they won't be included in step_files
                file_list = [f for f in file_list if f not in still_bad]
            else:
                print(f"[validation] All files validated successfully")
        
        self.step_files[step] = [self.local_cache / f for f in file_list]

    def prefetch_step_background(self, file_list, step: int):
        if self.rank != 0:
            return
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            self.prefetch_thread.join()
        print(f"[prefetch] Starting background download for step {step} ({len(file_list)} files)")
        self.prefetch_thread = threading.Thread(
            target=self.download_step_files,
            args=(file_list, step),
            daemon=True,
        )
        self.prefetch_step = step
        self.prefetch_thread.start()

    def cleanup_old_steps(self, current_step: int):
        if self.rank != 0:
            return
        min_keep = max(0, current_step - MAX_CACHE_STEPS + 1)
        # Only delete training steps (>= 0), never delete validation (-1) or calibration (-2) files
        to_delete = [s for s in self.step_files.keys() if s >= 0 and s < min_keep]
        if to_delete:
            total_files = sum(len(self.step_files[s]) for s in to_delete)
            print(f"[cleanup] Removing {len(to_delete)} old steps ({total_files} files)")
        for step in to_delete:
            for path in self.step_files[step]:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
            del self.step_files[step]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--calib-csv", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--mmap", action="store_true")
    parser.add_argument("--stream", action="store_true")
    # rclone parameters (auto-detected if not specified)
    parser.add_argument("--rclone-transfers", type=int, default=None, 
                        help="Parallel file transfers (auto-detect based on CPU/RAM)")
    parser.add_argument("--rclone-checkers", type=int, default=None,
                        help="Parallel checkers (auto-detect based on CPU/RAM)")
    parser.add_argument("--rclone-buffer", type=str, default=None,
                        help="Buffer size per transfer e.g. 64M, 128M (auto-detect)")
    parser.add_argument("--rclone-threads", type=int, default=None,
                        help="Multi-thread streams per file (auto-detect)")
    parser.add_argument("--epochs", type=float, default=1.2)
    parser.add_argument("--batch-size", type=int, default=4)  # Reduced from 8 for InternVideo2 teacher compatibility
    parser.add_argument("--samples-per-step", type=int, default=SAMPLES_PER_STEP)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.0, 
                        help="Label smoothing (0.0 = hard labels, better for Brier score)")
    parser.add_argument("--model-id", type=str, default="local:/home/shadeform/models/InternVideo2-stage2_1b-224p-f4.pt")
    parser.add_argument("--output-dir", type=str, default="checkpoints/videomae_v2")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--inference-temp", type=float, default=1.0)
    # Loss weights optimized for SN34 - AGGRESSIVE BRIER STRATEGY
    parser.add_argument("--bce-weight", type=float, default=0.0, help="BCE weight (0 = disabled, not part of SN34)")
    parser.add_argument("--distill-weight", type=float, default=0.4, help="Distillation weight (learn from 6B teacher)")
    parser.add_argument("--brier-weight", type=float, default=0.6, help="Brier weight (directly optimizes SN34 - 1.8x multiplier!)")
    parser.add_argument("--distill-temp", type=float, default=2.0)
    parser.add_argument("--distill-warmup-samples", type=int, default=300000)
    parser.add_argument("--distill-warmup-mode", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--teacher-model-id", type=str, default="/home/shadeform/models/InternVideo2-Stage2_6B-224p-f4/internvideo2-s2_6b-224p-f4_with_audio_encoder.pt")
    parser.add_argument(
        "--teacher-type",
        type=str,
        default="internvideo2_local",
        choices=["internvideo2", "videomae", "internvideo2_hf", "internvideo2_local"],
    )
    parser.add_argument("--teacher-repo", type=str, default="OpenGVLab/InternVideo")
    parser.add_argument("--teacher-entrypoint", type=str, default="InternVideo2-Stage2_6B-224p-f4")
    parser.add_argument("--teacher-input", type=str, default="raw", choices=["raw", "normalized"])
    parser.add_argument(
        "--internvideo2-code-path",
        type=str,
        default="~/InternVideo/InternVideo2/single_modality",
        help="Path to InternVideo2 repo single_modality folder for local teacher",
    )
    parser.add_argument("--distill-kind", type=str, default="feature", choices=["kl", "feature"])
    parser.add_argument("--feature-mse-weight", type=float, default=0.5, help="Weight for MSE in feature loss")
    parser.add_argument("--feature-cosine-weight", type=float, default=0.5, help="Weight for Cosine in feature loss")
    parser.add_argument("--teacher-frames", type=int, default=8, help="Sample N frames for teacher (default=8 for InternVideo2-6B)")
    parser.add_argument(
        "--calib-temps",
        type=str,
        default="0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5",
        help="Comma-separated temperatures for SN34 sweep",
    )
    parser.add_argument("--calib-metric", type=str, default="brier", choices=["brier", "sn34"])
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--nan-lr-scale", type=float, default=0.5)
    parser.add_argument("--sanity-save", action="store_true")
    parser.add_argument("--sanity-path", type=str, default="sanity_frame.png")
    parser.add_argument("--swa-start-samples", type=int, default=2700000)
    parser.add_argument("--swa-every", type=int, default=75000)
    parser.add_argument("--swa-max", type=int, default=8)
    parser.add_argument("--swa-apply", action="store_true")
    parser.add_argument("--temporal-skip-prob", type=float, default=0.2)
    parser.add_argument("--color-jitter", type=float, default=0.1)
    parser.add_argument("--color-jitter-prob", type=float, default=0.2)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=0.0,
                        help="Focal loss gamma. 0=standard CE, 2=strong hard-example focus. "
                             "Focal loss down-weights easy examples by (1-p_t)^gamma, focusing "
                             "gradient on hard examples (critical for per-dataset >95%% accuracy). "
                             "Recommended: 1.0-2.0 for multi-dataset training.")
    parser.add_argument("--jpeg-prob", type=float, default=0.0,
                        help="Probability of applying JPEG compression augmentation per video. "
                             "THE single most impactful augmentation for generalization to "
                             "unknown generators (+4.4%% AUROC, Pellegrini et al. 2025). "
                             "Recommended: 0.3 for training, 0.0 for fine-tuning.")
    parser.add_argument("--jpeg-quality-min", type=int, default=50,
                        help="Minimum JPEG quality for compression augmentation (lower=more artifacts).")
    parser.add_argument("--jpeg-quality-max", type=int, default=100,
                        help="Maximum JPEG quality for compression augmentation.")
    parser.add_argument("--gaussian-noise-prob", type=float, default=0.0,
                        help="Probability of adding Gaussian noise per video. "
                             "Forces the model to learn genuine artifact patterns, not noise. "
                             "Recommended: 0.15-0.25.")
    parser.add_argument("--gaussian-noise-std-min", type=float, default=0.005,
                        help="Minimum noise standard deviation.")
    parser.add_argument("--gaussian-noise-std-max", type=float, default=0.03,
                        help="Maximum noise standard deviation.")
    parser.add_argument("--gaussian-blur-prob", type=float, default=0.0,
                        help="Probability of applying Gaussian blur per video. "
                             "Improves robustness to blurring from video compression/scaling. "
                             "Recommended: 0.10-0.20.")
    parser.add_argument("--swa-min-sn34", type=float, default=0.85, help="Only collect SWA if validation SN34 >= this (quality threshold)")
    parser.add_argument("--grad-accum", type=int, default=2)  # Increased to maintain effective batch size with reduced per-GPU batch
    parser.add_argument("--eval-every-steps", type=int, default=3124)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--sn34-alpha", type=float, default=1.2)
    parser.add_argument("--sn34-beta", type=float, default=1.8)
    parser.add_argument("--h200-4x", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--reset-step", action="store_true", help="Reset step/epoch counters when resuming (for fine-tuning)")
    parser.add_argument("--backbone-lr-scale", type=float, default=1.0,
                        help="Backbone top-layer LR = lr * backbone_lr_scale (e.g. 0.01 = 100x lower)")
    parser.add_argument("--layer-lr-decay", type=float, default=1.0,
                        help="Per-layer LR decay factor for backbone (e.g. 0.75 = each deeper layer gets 0.75x LR)")
    parser.add_argument("--backbone-warmup-frac", type=float, default=0.0,
                        help="Fraction of training to keep backbone frozen (0.15 = first 15%% head-only, "
                             "then linearly ramp backbone LR to target over next 10%%). "
                             "Prevents destroying pretrained features.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint-mcc-floor", type=float, default=0.0,
                        help="Require MCC >= this value before updating best.pt by SN34.")
    parser.add_argument("--max-dummy-ratio", type=float, default=0.02,
                        help="Abort if black dummy-sample ratio exceeds this after warmup.")
    parser.add_argument("--dummy-ratio-warmup-samples", type=int, default=20000,
                        help="Minimum observed samples before enforcing --max-dummy-ratio.")
    parser.add_argument("--objective-schedule", type=str, default="three_stage", choices=["legacy", "three_stage"],
                        help="Dynamic objective schedule across training progress.")
    parser.add_argument("--stage1-frac", type=float, default=0.60,
                        help="End fraction for stage-1 (generalization) under three_stage schedule.")
    parser.add_argument("--stage2-frac", type=float, default=0.85,
                        help="End fraction for stage-2 (stabilization) under three_stage schedule.")
    parser.add_argument("--dataset-balance", action="store_true",
                        help="Use dataset-balanced sampling: each dataset gets equal gradient "
                             "contribution per step via WeightedRandomSampler, and the training "
                             "CSV is stratified-shuffled so every step block sees all datasets.")
    parser.add_argument("--macro-avg-eval", action="store_true",
                        help="Use macro-averaged SN34 for checkpoint selection. Each dataset "
                             "contributes equally to the val metric, preventing large datasets "
                             "from drowning out small ones.")
    parser.add_argument("--max-sample-repeat", type=float, default=20.0,
                        help="Max effective repeats per sample per epoch via weighted sampling. "
                             "Prevents memorization of tiny datasets. With value=50 and 3 epochs, "
                             "each sample seen at most ~150 times total. 0=no cap (dangerous).")
    parser.add_argument("--min-dataset-sn34", type=float, default=0.0,
                        help="Minimum per-dataset SN34 required to save a best checkpoint. "
                             "If any dataset's SN34 falls below this floor, the checkpoint is "
                             "rejected even if the macro SN34 improved. Prevents sacrificing "
                             "small datasets for overall improvement. Requires --macro-avg-eval. "
                             "Recommended: 0.15-0.25 for diverse multi-dataset training.")
    args = parser.parse_args()

    if args.h200_4x:
        args.batch_size = 4
        args.grad_accum = 4
        args.lr = 5e-5
        # Optimal rclone settings for 4xH200 (auto-detect based on actual CPUs)
        cpu_count = os.cpu_count() or 128
        if args.rclone_transfers is None:
            args.rclone_transfers = min(cpu_count, 256)  # Use all available CPUs, max 256
        if args.rclone_checkers is None:
            args.rclone_checkers = 0  # Disable checkers with --ignore-existing for max speed
        if args.rclone_buffer is None:
            args.rclone_buffer = "256M"
        if args.rclone_threads is None:
            # Scale threads based on transfers to avoid too many connections
            # With 176 transfers: 176 * 4 = 704 concurrent connections (reasonable)
            args.rclone_threads = 4 if args.rclone_transfers > 100 else 8

    if not (0.0 < args.stage1_frac < args.stage2_frac < 1.0):
        raise ValueError("--stage1-frac and --stage2-frac must satisfy 0 < stage1 < stage2 < 1")

    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")
    
    # Set seeds for reproducibility (different per rank for data variety)
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if is_main:
        print(f"[Seed] Base seed: {args.seed} (rank {rank} uses {seed})")
    
    # Standard VideoMAE format: [B, T, C, H, W]
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load VideoMAE-v2 (only rank0 downloads, others wait)
    if is_main:
        print(f"[Model] Loading VideoMAE-v2 from {args.model_id}")
    
    # Use custom wrapper for VideoMAEv2-giant (feature extraction model + classifier)
    if world_size > 1 and rank == 0:
        model = VideoMAEv2ForClassification(args.model_id, num_labels=2)
        dist.barrier()
    elif world_size > 1:
        dist.barrier()
        model = VideoMAEv2ForClassification(args.model_id, num_labels=2)
    else:
        model = VideoMAEv2ForClassification(args.model_id, num_labels=2)
    
    model = model.to(device)
    
    if is_main:
        param_count = sum(p.numel() for p in model.parameters())
        backbone_params = sum(p.numel() for p in model.videomae.parameters())
        classifier_params = param_count - backbone_params
        print(f"[Model] Total parameters: {param_count/1e6:.1f}M")
        print(f"[Model] Backbone: {backbone_params/1e6:.1f}M, Classifier: {classifier_params/1e3:.1f}K")
        print(f"[Model] Estimated BF16 size: {param_count * 2 / (1024**3):.2f} GB")
        print(f"[Model] Device: {device}")
        
        # Verify architecture is Giant (should be ~1000M params)
        if backbone_params < 500e6:
            print(f"⚠️  WARNING: Expected ~1B params for Giant, got {backbone_params/1e6:.1f}M")
            print(f"⚠️  You may be loading the wrong model size!")
        
        # Verify positional embedding keys
        pos_embed_keys = [k for k in model.state_dict().keys() 
                         if any(p in k.lower() for p in ["pos_embed", "position", "temporal"])]
        if pos_embed_keys:
            print(f"[Model] Position embedding keys: {pos_embed_keys[:5]}...")  # First 5

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        if is_main:
            print(f"[DDP] Initialized with {world_size} GPUs")

    teacher = load_teacher(args, device, is_main)
    distill_kind = args.distill_kind
    if teacher is not None and args.teacher_type == "internvideo2_hf":
        distill_kind = "feature"
    
    # Verify teacher architecture
    if teacher is not None and is_main:
        teacher_params = sum(p.numel() for p in teacher.parameters())
        print(f"[Teacher] Parameters: {teacher_params/1e9:.2f}B")
        # Check for temporal position embeddings
        teacher_pos_keys = [k for k in teacher.state_dict().keys()
                          if any(p in k.lower() for p in ["pos_embed", "temporal_pos", "time_embed", "rel_pos"])]
        if teacher_pos_keys:
            print(f"[Teacher] Position embedding keys: {teacher_pos_keys[:5]}...")

    data_manager = BackblazeDataManager(
        rank=rank,
        rclone_transfers=args.rclone_transfers,
        rclone_checkers=args.rclone_checkers,
        rclone_buffer=args.rclone_buffer,
        rclone_threads=args.rclone_threads,
    ) if args.stream else None

    # Load validation
    if args.stream:
        val_df = pd.read_csv(args.val_csv)
        val_files = [row["bgr_path"] for _, row in val_df.iterrows()]
        if is_main:
            data_manager.download_step_files(val_files, -1)
        if world_size > 1:
            dist.barrier()
        val_dataset = VideoNpyDataset(args.val_csv, data_manager=data_manager, mmap=args.mmap)
        val_dataset.df = val_df
    else:
        val_dataset = VideoNpyDataset(args.val_csv, data_root=args.data_root, mmap=args.mmap)

    # Scale num_workers by world_size to prevent CPU oversubscription
    cpu_count = os.cpu_count() or 8
    num_workers = min(cpu_count // max(1, world_size), 16)  # Max 16 per GPU
    num_workers = max(4, num_workers)  # Min 4 workers
    # Enable macro-averaged evaluation (per-dataset SN34 → equal-weight average)
    dataset_idx_to_name = None
    if args.macro_avg_eval:
        ds_map = val_dataset.enable_dataset_tracking()
        dataset_idx_to_name = {v: k for k, v in ds_map.items()}
        if is_main:
            print(f"[Eval] Macro-average enabled: {len(ds_map)} datasets tracked")

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=(val_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
    )
    calib_loader = val_loader
    
    if args.calib_csv:
        if args.stream:
            calib_df = pd.read_csv(args.calib_csv)
            calib_files = [row["bgr_path"] for _, row in calib_df.iterrows()]
            if is_main:
                data_manager.download_step_files(calib_files, -2)
            if world_size > 1:
                dist.barrier()
            calib_dataset = VideoNpyDataset(args.calib_csv, data_manager=data_manager, mmap=args.mmap)
            calib_dataset.df = calib_df
        else:
            calib_dataset = VideoNpyDataset(args.calib_csv, data_root=args.data_root, mmap=args.mmap)
        if args.macro_avg_eval:
            calib_dataset.enable_dataset_tracking()
        calib_sampler = DistributedSampler(calib_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
        calib_loader = DataLoader(
            calib_dataset,
            batch_size=args.batch_size,
            sampler=calib_sampler,
            shuffle=(calib_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
        )

    # Build optimizer with layer-wise LR decay
    if args.backbone_lr_scale < 1.0 or args.layer_lr_decay < 1.0:
        if is_main:
            print(f"\n[Optimizer] Layer-wise LR: head={args.lr:.2e}, backbone_scale={args.backbone_lr_scale}, layer_decay={args.layer_lr_decay}")
        
        base_model = model.module if world_size > 1 else model
        param_groups = []

        def _extract_block_index(param_name: str):
            prefixes = (
                "videomae.model.blocks.",
                "videomae.backbone.blocks.",
                "videomae.blocks.",
            )
            for prefix in prefixes:
                if prefix in param_name:
                    suffix = param_name.split(prefix, 1)[1]
                    tok = suffix.split(".", 1)[0]
                    if tok.isdigit():
                        return int(tok), prefix
            return None, None
        
        # Head params (fc_norm + classifier) get full LR
        head_params = []
        head_names = []
        for name, param in base_model.named_parameters():
            if 'fc_norm' in name or 'classifier' in name:
                head_params.append(param)
                head_names.append(name)
        param_groups.append({
            "params": head_params,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "name": "head",
        })
        if is_main:
            print(f"  head ({len(head_params)} params): lr={args.lr:.2e}")
        
        # Find number of blocks
        num_blocks = 0
        block_prefix_used = None
        for name, _ in base_model.named_parameters():
            block_num, prefix = _extract_block_index(name)
            if block_num is not None:
                num_blocks = max(num_blocks, block_num + 1)
                if block_prefix_used is None:
                    block_prefix_used = prefix
        
        # Backbone block params with layer-wise decay
        # Top block (num_blocks-1) gets backbone_lr_scale * lr
        # Each deeper block gets layer_lr_decay times the one above
        assigned_params = set(id(p) for p in head_params)
        
        for block_idx in range(num_blocks - 1, -1, -1):
            depth_from_top = (num_blocks - 1) - block_idx
            block_lr = args.lr * args.backbone_lr_scale * (args.layer_lr_decay ** depth_from_top)
            
            block_params = []
            for name, param in base_model.named_parameters():
                block_num, _ = _extract_block_index(name)
                if block_num == block_idx and id(param) not in assigned_params:
                    block_params.append(param)
                    assigned_params.add(id(param))
            
            if block_params:
                param_groups.append({
                    "params": block_params,
                    "lr": block_lr,
                    "weight_decay": args.weight_decay,
                    "name": f"block_{block_idx}",
                })
        
        if is_main:
            if num_blocks > 0:
                top_block_lr = args.lr * args.backbone_lr_scale
                bottom_block_lr = args.lr * args.backbone_lr_scale * (args.layer_lr_decay ** (num_blocks - 1))
                print(f"  blocks ({num_blocks} layers @ {block_prefix_used}): top_lr={top_block_lr:.2e}, bottom_lr={bottom_block_lr:.2e}")
            else:
                print("  blocks (0 layers detected): using backbone_other group only")
        
        # Remaining backbone params (patch_embed, norm, pos_embed, etc.) get lowest LR
        remaining_params = []
        for name, param in base_model.named_parameters():
            if id(param) not in assigned_params:
                remaining_params.append(param)
                assigned_params.add(id(param))
        
        if remaining_params:
            lowest_lr = args.lr * args.backbone_lr_scale * (args.layer_lr_decay ** num_blocks)
            param_groups.append({
                "params": remaining_params,
                "lr": lowest_lr,
                "weight_decay": args.weight_decay,
                "name": "backbone_other",
            })
            if is_main:
                print(f"  backbone_other ({len(remaining_params)} params): lr={lowest_lr:.2e}")
        
        total_assigned = sum(len(pg["params"]) for pg in param_groups)
        total_model = sum(1 for _ in base_model.parameters())
        if is_main:
            print(f"  Total param groups: {len(param_groups)}, params: {total_assigned}/{total_model}")
        
        optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    use_bf16 = not args.no_bf16
    scaler = None if use_bf16 else GradScaler()

    if is_main:
        print(f"\n[Dataset] Loading training CSV: {args.train_csv}")
    train_df = pd.read_csv(args.train_csv)

    if args.dataset_balance and "dataset" in train_df.columns:
        if is_main:
            print(f"[Dataset] Stratified shuffle (seed={args.seed}) — all datasets in every step block...")
        train_df = stratified_shuffle_df(train_df, seed=args.seed)
        sample_weights = compute_dataset_weights(train_df, max_repeat=args.max_sample_repeat)
        if is_main:
            ds_counts = train_df["dataset"].value_counts()
            print(f"[Dataset] {len(ds_counts)} datasets, weight range: "
                  f"{sample_weights.min():.4f} – {sample_weights.max():.4f} "
                  f"(max_repeat={args.max_sample_repeat})")
    else:
        # Standard random shuffle
        if is_main:
            print(f"[Dataset] Shuffling training data (seed={args.seed})...")
        train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        sample_weights = None
    
    # Smoke test mode
    smoke_max_samples = None
    if args.smoke_steps > 0:
        smoke_max_samples = args.batch_size * max(1, args.grad_accum) * max(1, args.smoke_steps) * max(1, world_size)
        train_df = train_df.iloc[:min(len(train_df), smoke_max_samples)]
        if is_main:
            print(f"[Smoke test] Limited to {len(train_df)} samples ({args.smoke_steps} steps)")
    
    total_samples = len(train_df)
    if is_main:
        print(f"[Dataset] Training samples: {total_samples:,}")
        print(f"[Dataset] CPU workers: {num_workers}")
    samples_per_step = max(1, args.samples_per_step)
    if smoke_max_samples is not None:
        samples_per_step = max(1, total_samples)
    num_steps = (total_samples + samples_per_step - 1) // samples_per_step

    # NOTE:
    # `step` is incremented on optimizer updates (after grad accumulation), not per micro-batch.
    # LR schedule must therefore be based on optimizer-step count.
    micro_steps_per_epoch = int(math.ceil(total_samples / (args.batch_size * world_size)))
    optimizer_steps_per_epoch = int(math.ceil(micro_steps_per_epoch / max(1, args.grad_accum)))
    total_steps = int(math.ceil(optimizer_steps_per_epoch * args.epochs))
    warmup_steps = int(total_steps * args.warmup_frac)
    
    if is_main:
        print(f"\n[Training] Config:")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Gradient accumulation: {args.grad_accum}")
        print(f"  Effective batch size: {args.batch_size * world_size * args.grad_accum}")
        print(f"  Micro-steps/epoch: {micro_steps_per_epoch:,}")
        print(f"  Optimizer-steps/epoch: {optimizer_steps_per_epoch:,}")
        print(f"  Learning rate: {args.lr:.2e}")
        print(f"  Label smoothing: {args.label_smoothing}")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Warmup steps: {warmup_steps:,}")
        if micro_steps_per_epoch % max(1, args.grad_accum) != 0:
            dropped = micro_steps_per_epoch % max(1, args.grad_accum)
            print(
                f"  ⚠️ Micro-step remainder per epoch: {dropped} "
                f"(not divisible by grad_accum={args.grad_accum}); "
                "last partial accumulation in each step chunk may be skipped."
            )
        print(f"  Streaming: {args.stream}")
        print(f"  SWA start: {args.swa_start_samples:,} samples")
        print(f"  Loss weights: Brier={args.brier_weight} Distill={args.distill_weight} BCE={args.bce_weight}")
        if args.bce_weight == 0:
            print(f"  📊 SN34-focused training (BCE disabled)")
        if args.label_smoothing > 0.0 and args.brier_weight > 0:
            print(f"  ⚠️ Warning: label_smoothing={args.label_smoothing} conflicts with Brier loss (use 0.0 for max SN34)")
        if args.label_smoothing == 0.0:
            print(f"  ✓ Hard labels (optimal for Brier score)")
        print(f"  Augmentations: hflip={args.hflip_prob} temporal_skip={args.temporal_skip_prob} color_jitter={args.color_jitter}")
        if teacher is None:
            print("  Teacher: disabled")
        else:
            print(f"  Teacher: {args.teacher_type} ({args.teacher_model_id})")
            print(f"  Distill: {distill_kind} (warmup={args.distill_warmup_samples})")
            if distill_kind == "feature":
                print(f"  Feature loss: MSE={args.feature_mse_weight} Cosine={args.feature_cosine_weight}")
                print(f"  Note: MSE is normalized by teacher_dim for stable scaling")
            if args.teacher_frames > 0:
                print(f"  Teacher frames: {args.teacher_frames} (sampled from 16)")
        if args.swa_min_sn34 > 0:
            print(f"  SWA quality threshold: SN34 >= {args.swa_min_sn34}")
        print(f"  Checkpoint MCC floor: {args.checkpoint_mcc_floor:.3f}")
        print(f"  Dummy guard: max_ratio={args.max_dummy_ratio:.3%} after {args.dummy_ratio_warmup_samples:,} samples")
        print(f"  Objective schedule: {args.objective_schedule} (stage1={args.stage1_frac:.2f}, stage2={args.stage2_frac:.2f})")
        
        # Final workflow verification
        print(f"\n[Workflow Verification]")
        print(f"  ✓ Color space: BGR → RGB (in VideoNpyDataset)")
        print(f"  ✓ Precision: {'BF16' if use_bf16 else 'FP32'}")
        if args.bce_weight > 0:
            print(f"  ✓ Loss: Brier({args.brier_weight}) + Distill({args.distill_weight}) + BCE({args.bce_weight})")
        else:
            print(f"  ✓ Loss: Brier({args.brier_weight}) + Distill({args.distill_weight}) [SN34-optimized]")
        print(f"  ✓ Gradient clipping: max_norm={args.clip_grad}")
        print(f"  ✓ Validation: val_csv for selection, calib_csv for T-calibration")
        if args.calib_csv:
            print(f"  ✓ Separate calibration set: {args.calib_csv}")

    # Store initial LR ratios for layer-wise scheduling
    initial_lrs = [pg["lr"] for pg in optimizer.param_groups]
    pg_names = [pg.get("name", "") for pg in optimizer.param_groups]

    # Backbone warmup: freeze backbone for first N% of training, then linearly
    # ramp up over the next 10% to the target backbone LR.
    # This lets the classifier head learn a good starting point before the
    # backbone starts adapting, preventing destruction of pretrained features.
    backbone_freeze_steps = int(total_steps * args.backbone_warmup_frac)
    backbone_ramp_steps = int(total_steps * 0.10)  # 10% linear ramp after freeze
    if args.backbone_warmup_frac > 0 and is_main:
        print(f"\n[Backbone Warmup] Freeze for {backbone_freeze_steps} steps "
              f"({args.backbone_warmup_frac:.0%}), ramp over {backbone_ramp_steps} steps")

    def get_lr_scale(step):
        """Returns a multiplier [0, 1] for cosine schedule with warmup."""
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def get_backbone_scale(step):
        """Returns 0→1 multiplier for backbone LR (freeze then ramp)."""
        if backbone_freeze_steps <= 0:
            return 1.0
        if step < backbone_freeze_steps:
            return 0.0  # Frozen
        ramp_progress = (step - backbone_freeze_steps) / max(1, backbone_ramp_steps)
        return min(1.0, ramp_progress)  # Linear ramp 0→1

    step = 0
    global_samples_processed = 0
    best_sn34 = 0.0
    start_epoch = 0
    epochs_int = math.floor(args.epochs)
    extra_frac = args.epochs - epochs_int

    swa_state = None
    swa_count = 0
    next_swa = args.swa_start_samples
    running_loss = 0.0
    running_bce = 0.0
    running_brier = 0.0
    running_kl = 0.0
    running_feat = 0.0
    running_steps = 0
    dummy_samples_seen = 0
    total_samples_seen = 0
    last_stage_name = None
    swa_phase_announced = False
    saved_proj_state = None  # Will be set if resuming from checkpoint with projector
    
    # Try to resume from checkpoint
    resume_path = Path(args.output_dir) / "best.pt"
    if resume_path.exists():
        if is_main:
            print(f"\n{'='*60}")
            print(f"RESUMING FROM CHECKPOINT: {resume_path}")
            print(f"{'='*60}")
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model_to_load = model.module if world_size > 1 else model
            model_to_load.load_state_dict(checkpoint["model_state_dict"])
            
            # Load optimizer if available (for new checkpoints)
            if "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    if is_main:
                        print(f"[Resume] Loaded optimizer state")
                except Exception as e:
                    if is_main:
                        print(f"[Resume] Warning: Failed to load optimizer state: {e}")
            
            step = checkpoint.get("step", 0)
            best_sn34 = checkpoint.get("best_sn34", 0.0)
            swa_count = checkpoint.get("swa_count", 0)
            
            # Load projector if saved (critical for feature distillation!)
            if "projector_state_dict" in checkpoint and checkpoint["projector_state_dict"] is not None:
                # Projector will be created on first batch, we'll load it then
                saved_proj_state = checkpoint["projector_state_dict"]
                if is_main:
                    print(f"[Resume] Found projector state (will load after initialization)")
            else:
                saved_proj_state = None
            
            # Reset step/epoch for fine-tuning
            if args.reset_step:
                if is_main:
                    print(f"[Resume] --reset-step: Resetting step={step}->0, best_sn34={best_sn34:.4f}->0, swa_count={swa_count}->0")
                step = 0
                best_sn34 = 0.0
                swa_count = 0
                start_epoch = 0
                saved_proj_state = None  # Don't load old projector for fine-tuning
            
            # Calculate which epoch we're in
            if step > 0:
                start_epoch = step // num_steps
            if is_main:
                print(f"[Resume] Loaded checkpoint:")
                print(f"  Model weights: ✓")
                print(f"  Step: {step}")
                print(f"  Best SN34: {best_sn34:.4f}")
                print(f"  SWA count: {swa_count}")
                if saved_proj_state is not None:
                    print(f"  Projector: ✓ (will load)")
                if step > 0:
                    print(f"  Starting from epoch: {start_epoch + 1}")
                else:
                    print(f"  Fine-tuning from epoch 0" if args.reset_step else f"  Note: Old checkpoint (no step info), will retrain but keep best model")
        except Exception as e:
            if is_main:
                print(f"[Resume] Failed to load checkpoint: {e}")
                print(f"[Resume] Starting from scratch...")
    elif is_main:
        print(f"\n[Resume] No checkpoint found, starting from scratch...")
    
    # Initialize projector BEFORE training loop to avoid DDP issues
    # We'll do a dummy forward pass to get dimensions
    proj = None
    proj_initialized = False
    # saved_proj_state is already set during checkpoint loading (don't reset!)

    for epoch in range(start_epoch, epochs_int + (1 if extra_frac > 0 else 0)):
        if is_main:
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{epochs_int + (1 if extra_frac > 0 else 0)}")
            if epoch == start_epoch and start_epoch > 0:
                print(f"(RESUMED from step {step})")
            print(f"{'='*60}")
        
        model.train()
        max_steps_this_epoch = int(num_steps * (extra_frac if epoch == epochs_int and extra_frac > 0 else 1.0))
        
        # Skip already completed steps when resuming
        start_step_idx = (step % num_steps) if epoch == start_epoch else 0

        for step_idx in range(start_step_idx, num_steps):
            if step_idx >= max_steps_this_epoch:
                break
            global_step = epoch * num_steps + step_idx
            start_idx = step_idx * samples_per_step
            end_idx = min((step_idx + 1) * samples_per_step, total_samples)
            step_df = train_df.iloc[start_idx:end_idx]

            total_target_samples = max(1, int(total_samples * max(args.epochs, 1.0)))
            train_progress = min(1.0, global_samples_processed / total_target_samples)
            stage_name = "legacy"
            cur_temporal_skip_prob = args.temporal_skip_prob
            cur_hflip_prob = args.hflip_prob
            cur_color_jitter_prob = args.color_jitter_prob
            cur_color_jitter = args.color_jitter
            cur_aug_mult = 1.0
            bce_stage_mult = 1.0
            brier_stage_mult = 1.0
            distill_stage_mult = 1.0
            if args.objective_schedule == "three_stage":
                if train_progress < args.stage1_frac:
                    stage_name = "stage1-generalize"
                    bce_stage_mult = 1.00
                    brier_stage_mult = 0.85
                    distill_stage_mult = 1.00
                    cur_aug_mult = 1.0
                elif train_progress < args.stage2_frac:
                    stage_name = "stage2-stabilize"
                    bce_stage_mult = 0.75
                    brier_stage_mult = 1.00
                    distill_stage_mult = 0.80
                    cur_temporal_skip_prob = args.temporal_skip_prob * 0.50
                    cur_hflip_prob = args.hflip_prob * 0.50
                    cur_color_jitter_prob = args.color_jitter_prob * 0.50
                    cur_color_jitter = args.color_jitter * 0.75
                    cur_aug_mult = 0.5
                else:
                    stage_name = "stage3-sn34"
                    bce_stage_mult = 0.50
                    brier_stage_mult = 1.15
                    distill_stage_mult = 0.60
                    cur_temporal_skip_prob = 0.0
                    cur_hflip_prob = 0.0
                    cur_color_jitter_prob = 0.0
                    cur_color_jitter = 0.0
                    cur_aug_mult = 0.0
            if is_main and stage_name != last_stage_name:
                print(
                    f"[Schedule] {stage_name} progress={train_progress:.1%} "
                    f"aug(skip={cur_temporal_skip_prob:.2f},hflip={cur_hflip_prob:.2f},cj_prob={cur_color_jitter_prob:.2f}) "
                    f"loss_mult(bce={bce_stage_mult:.2f},brier={brier_stage_mult:.2f},distill={distill_stage_mult:.2f})"
                )
                last_stage_name = stage_name

            if args.stream:
                step_files = [row["bgr_path"] for _, row in step_df.iterrows()]
                if is_main:
                    try:
                        data_manager.download_step_files(step_files, global_step)
                    except Exception as e:
                        print(f"[ERROR] Download failed for step {global_step}: {e}")
                        print(f"[ERROR] Will try to use cached files or skip bad samples...")
                    
                    # Prefetch next step (non-blocking)
                    try:
                        if step_idx < num_steps - 1:
                            next_start = (step_idx + 1) * samples_per_step
                            next_end = min((step_idx + 2) * samples_per_step, total_samples)
                            next_df = train_df.iloc[next_start:next_end]
                            next_files = [row["bgr_path"] for _, row in next_df.iterrows()]
                            data_manager.prefetch_step_background(next_files, global_step + 1)
                    except Exception as e:
                        print(f"[ERROR] Prefetch failed: {e}")
                if world_size > 1:
                    # Streaming downloads can be long; make barrier device-aware to avoid NCCL rank/device mismatch.
                    dist.barrier(device_ids=[local_rank])

                train_dataset = VideoNpyDataset(
                    args.train_csv,
                    data_manager=data_manager,
                    mmap=args.mmap,
                    temporal_skip_prob=cur_temporal_skip_prob,
                )
                train_dataset.df = step_df
            else:
                train_dataset = VideoNpyDataset(
                    args.train_csv,
                    data_root=args.data_root,
                    mmap=args.mmap,
                    temporal_skip_prob=cur_temporal_skip_prob,
                )
                train_dataset.df = step_df

            # Build per-step sampler: weighted (dataset-balanced) or standard
            if args.dataset_balance and sample_weights is not None and "dataset" in step_df.columns:
                # Extract weights for this step's samples from the global weight array
                step_global_indices = step_df.index.tolist() if step_df.index.max() < len(sample_weights) else list(range(len(step_df)))
                try:
                    step_w = sample_weights[step_global_indices]
                except (IndexError, KeyError):
                    step_w = compute_dataset_weights(step_df, max_repeat=args.max_sample_repeat)
                step_w_tensor = torch.from_numpy(step_w).double()

                if world_size > 1:
                    # DDP: all ranks must generate the SAME weighted indices for clean partitioning.
                    # Seed the generator deterministically per step so all ranks agree.
                    n = len(step_df)
                    ws_gen = torch.Generator()
                    ws_gen.manual_seed(args.seed * 100003 + epoch * num_steps + step_idx)
                    all_indices = list(WeightedRandomSampler(step_w_tensor, n, replacement=True, generator=ws_gen))
                    # Partition across ranks (deterministic, disjoint)
                    rank_indices = all_indices[rank::world_size]
                    train_sampler = torch.utils.data.SubsetRandomSampler(rank_indices)
                else:
                    train_sampler = WeightedRandomSampler(step_w_tensor, len(step_df), replacement=True)
            elif world_size > 1:
                train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
                train_sampler.set_epoch(epoch * num_steps + step_idx)
            else:
                train_sampler = None

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                shuffle=(train_sampler is None),
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=2,
            )

            grad_accum = max(1, args.grad_accum)
            optimizer.zero_grad()

            for batch_idx, (videos, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Step {step_idx+1}", leave=False)):
                batch_dummy = int((videos.view(videos.shape[0], -1).sum(dim=1) == 0).sum().item())
                dummy_samples_seen += batch_dummy
                total_samples_seen += int(videos.shape[0])
                # Track real samples seen (across all ranks) for stage schedule/distill warmup/SWA.
                # This must be independent of optimizer step frequency (grad accumulation).
                global_samples_processed += labels.numel() * world_size

                videos = videos.to(device, non_blocking=True).float() / 255.0  # [B, T, C, H, W]
                # Augmentations (on raw RGB [0,1]) - applied per-video independently
                B = videos.shape[0]
                if cur_hflip_prob > 0:
                    flip_mask = torch.rand(B) < cur_hflip_prob
                    for i in range(B):
                        if flip_mask[i]:
                            videos[i] = videos[i].flip(dims=[-1])
                if cur_color_jitter > 0 and cur_color_jitter_prob > 0:
                    jitter_mask = torch.rand(B) < cur_color_jitter_prob
                    for i in range(B):
                        if jitter_mask[i]:
                            videos[i] = apply_color_jitter(videos[i], cur_color_jitter)

                # Robustness augmentations — applied on raw [0,1] before normalization.
                # JPEG compression is the single most impactful augmentation for
                # generalization to unknown generators (+4.4% AUROC per research).
                # Stage multiplier (cur_aug_mult) reduces these in later training stages.
                if args.jpeg_prob > 0:
                    videos = apply_jpeg_augmentation(
                        videos, prob=args.jpeg_prob * cur_aug_mult,
                        quality_min=args.jpeg_quality_min,
                        quality_max=args.jpeg_quality_max,
                    )
                if args.gaussian_noise_prob > 0:
                    videos = apply_gaussian_noise(
                        videos, prob=args.gaussian_noise_prob * cur_aug_mult,
                        std_range=(args.gaussian_noise_std_min, args.gaussian_noise_std_max),
                    )
                if args.gaussian_blur_prob > 0:
                    videos = apply_gaussian_blur(
                        videos, prob=args.gaussian_blur_prob * cur_aug_mult,
                    )

                # Save raw videos for teacher BEFORE normalization (clone only if teacher needs raw)
                raw_videos = videos.clone() if (teacher is not None and args.teacher_input == "raw") else None
                videos = (videos - mean) / std  # Normalized for student [B, T, C, H, W]
                labels = labels.to(device, non_blocking=True)
                
                if args.sanity_save and is_main and step == 0 and batch_idx == 0:
                    # videos is [B, T, C, H, W], get first frame [C, H, W]
                    frame = videos[0, 0].detach().cpu()
                    frame = frame * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
                    frame = torch.clamp(frame, 0, 1)
                    torchvision.utils.save_image(frame, args.sanity_path)

                try:
                    with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        # Single forward pass - get both logits and hidden states
                        core = model.module if isinstance(model, DDP) else model
                        
                        # VideoMAE expects [B, C, T, H, W], we have [B, T, C, H, W]
                        videos_transposed = videos.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
                        
                        # Get encoder outputs (VideoMAEv2 doesn't accept return_dict arg)
                        encoder_outputs = core.videomae(pixel_values=videos_transposed)
                        
                        # Extract hidden state
                        if hasattr(encoder_outputs, 'last_hidden_state'):
                            hidden_state = encoder_outputs.last_hidden_state  # [B, num_patches, hidden_dim]
                        else:
                            hidden_state = encoder_outputs  # Already tensor
                        
                        # Match VideoMAEForVideoClassification forward pass exactly:
                        # 1. Mean pool, 2. Apply fc_norm if exists, 3. Classifier
                        if hidden_state.dim() == 3:
                            sequence_output = hidden_state.mean(dim=1)
                        else:
                            sequence_output = hidden_state
                        
                        if hasattr(core, 'fc_norm') and core.fc_norm is not None:
                            sequence_output = core.fc_norm(sequence_output)
                        logits = core.classifier(sequence_output).float()
                        probs = F.softmax(logits, dim=1)
                        
                        # Brier loss - directly optimizes SN34
                        brier_loss = F.mse_loss(probs[:, 1], labels.float())
                        
                        # BCE/Focal loss (optional - only compute if weight > 0)
                        if args.bce_weight > 0:
                            if args.focal_gamma > 0:
                                ce_loss = focal_cross_entropy(
                                    logits, labels, gamma=args.focal_gamma,
                                    label_smoothing=args.label_smoothing,
                                )
                            else:
                                ce_loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
                        else:
                            ce_loss = torch.tensor(0.0, device=logits.device)
                        distill_loss = torch.tensor(0.0, device=logits.device)
                        if teacher is not None and args.distill_weight > 0:
                            t_input = raw_videos if (args.teacher_input == "raw" and raw_videos is not None) else videos
                            if distill_kind == "kl":
                                # KL divergence on logits (for classifier teachers)
                                with torch.no_grad():
                                    t_out = teacher(t_input)
                                    t_logits = extract_logits(t_out).float()
                                T = max(args.distill_temp, 1e-6)
                                distill_loss = F.kl_div(
                                    F.log_softmax(logits / T, dim=1),
                                    F.softmax(t_logits / T, dim=1),
                                    reduction="batchmean",
                                ) * (T * T)
                            else:
                                # Feature distillation (for foundation model teachers like InternVideo2-6B)
                                # Optionally sample fewer frames for teacher (f4/f8 alignment)
                                t_input_sampled = sample_frames_for_teacher(t_input, args.teacher_frames)
                                
                                # Ensure teacher is in eval mode and no gradients
                                teacher.eval()
                                with torch.no_grad():
                                    t_feat = get_teacher_features(teacher, t_input_sampled, args.teacher_type)
                                    t_feat = t_feat.float().detach()  # Ensure detached from graph
                                
                                # Reuse sequence_output from first forward pass (no double forward!)
                                # This is the pooled + fc_norm output, same as what classifier sees
                                s_feat = sequence_output
                                
                                # Dimension validation and projector initialization
                                s_dim = s_feat.shape[-1]
                                t_dim = t_feat.shape[-1]
                                
                                if not proj_initialized:
                                    if is_main:
                                        print(f"[Projector] Initializing: student_dim={s_dim} -> teacher_dim={t_dim}")
                                    proj = nn.Linear(s_dim, t_dim, bias=False).to(device)
                                    
                                    # Load saved projector state if resuming
                                    if saved_proj_state is not None:
                                        try:
                                            proj.load_state_dict(saved_proj_state)
                                            if is_main:
                                                print(f"[Projector] ✓ Loaded from checkpoint (no random reset!)")
                                        except Exception as e:
                                            if is_main:
                                                print(f"[Projector] Warning: Failed to load saved state: {e}")
                                                print(f"[Projector] Using random initialization")
                                            nn.init.xavier_uniform_(proj.weight)
                                    else:
                                        # Initialize with small weights for stable training
                                        nn.init.xavier_uniform_(proj.weight)
                                    
                                    # DDP: Broadcast projector weights from rank 0 to all ranks
                                    if world_size > 1 and dist.is_initialized():
                                        for p in proj.parameters():
                                            dist.broadcast(p.data, src=0)
                                    
                                    optimizer.add_param_group({"params": proj.parameters(), "lr": args.lr, "name": "projector"})
                                    initial_lrs.append(args.lr)
                                    pg_names.append("projector")
                                    proj_initialized = True
                                
                                s_proj = proj(s_feat)
                                
                                # Combined loss: MSE (magnitude) + Cosine (direction)
                                # Normalize MSE by feature dimension to stabilize scale
                                mse_loss = F.mse_loss(s_proj, t_feat) / t_dim
                                cos_loss = cosine_similarity_loss(s_proj, t_feat)
                                distill_loss = (
                                    args.feature_mse_weight * mse_loss
                                    + args.feature_cosine_weight * cos_loss
                                )
                    distill_weight = args.distill_weight
                    if args.distill_warmup_samples > 0:
                        if args.distill_warmup_mode == "linear":
                            warmup_frac = min(1.0, global_samples_processed / args.distill_warmup_samples)
                        else:
                            warmup_frac = min(1.0, global_samples_processed / args.distill_warmup_samples)
                            warmup_frac = 0.5 * (1.0 - math.cos(math.pi * warmup_frac))
                        distill_weight = distill_weight * warmup_frac
                except Exception as forward_err:
                    print(f"[ERROR] Forward pass failed for batch {batch_idx}: {forward_err}")
                    print(f"[ERROR] Skipping batch...")
                    optimizer.zero_grad()  # Clear any partial gradients
                    continue  # Skip to next batch
                
                # Dynamic loss weights: schedule by training stage.
                bce_w = args.bce_weight * bce_stage_mult
                brier_w = args.brier_weight * brier_stage_mult
                distill_weight = distill_weight * distill_stage_mult
                if args.swa_start_samples > 0 and global_samples_processed >= args.swa_start_samples:
                    # SWA phase: MAXIMUM Brier focus (1.8x multiplier in SN34!)
                    if args.bce_weight == 0.0:  # If BCE was disabled, enable it now
                        if is_main and not swa_phase_announced:
                            print(f"\n{'='*60}")
                            print(f"[SWA PHASE] AGGRESSIVE BRIER MODE (SN34 = 1.0×MCC + 1.8×Brier)")
                            print(f"  BCE: 0.0 → 0.2 (minimal accuracy signal)")
                            print(f"  Brier: 0.6 → 0.5 (MAXIMUM calibration focus)")
                            print(f"  Distill: 0.4 → 0.3 (reduce teacher, model is mature)")
                            print(f"{'='*60}\n")
                            swa_phase_announced = True
                        bce_w = 0.2
                        brier_w = 0.5
                        distill_weight = distill_weight * 0.75  # Reduce distill from 0.4 to ~0.3
                
                loss = (
                    bce_w * ce_loss
                    + brier_w * brier_loss
                    + distill_weight * distill_loss
                ) / grad_accum

                if torch.isnan(loss):
                    for pg in optimizer.param_groups:
                        pg["lr"] *= args.nan_lr_scale
                    continue

                try:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                except Exception as backward_err:
                    print(f"[ERROR] Backward pass failed for batch {batch_idx}: {backward_err}")
                    print(f"[ERROR] Skipping batch...")
                    optimizer.zero_grad()
                    continue

                if (batch_idx + 1) % grad_accum == 0:
                    try:
                        # DDP: Manually sync projector gradients across GPUs (only before optimizer step!)
                        if proj is not None and world_size > 1 and dist.is_initialized():
                            for p in proj.parameters():
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                                    p.grad /= world_size
                        lr_scale = get_lr_scale(step)
                        bb_scale = get_backbone_scale(step)
                        for pg, init_lr, pg_name in zip(optimizer.param_groups, initial_lrs, pg_names):
                            is_backbone = pg_name.startswith("block_") or pg_name == "backbone_other"
                            pg["lr"] = init_lr * lr_scale * (bb_scale if is_backbone else 1.0)

                        # Collect all parameters for gradient clipping (model + projector)
                        all_params = list(model.parameters())
                        if proj is not None:
                            all_params.extend(list(proj.parameters()))
                        
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.clip_grad)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.clip_grad)
                            optimizer.step()

                        optimizer.zero_grad()
                        step += 1
                        
                        # global_samples_processed is updated per micro-batch above (real samples seen).
                    except Exception as optim_err:
                        print(f"[ERROR] Optimizer step failed for batch {batch_idx}: {optim_err}")
                        print(f"[ERROR] Clearing gradients and continuing...")
                        optimizer.zero_grad()
                        if scaler is not None:
                            scaler.update()  # Update scaler state
                        continue
                    running_loss += loss.item() * grad_accum
                    running_bce += ce_loss.item()
                    running_brier += brier_loss.item()
                    if teacher is not None:
                        if distill_kind == "kl":
                            running_kl += distill_loss.item()
                        else:
                            running_feat += distill_loss.item()
                    running_steps += 1

                    if is_main and args.log_every > 0 and (step % args.log_every == 0):
                        lr_now = optimizer.param_groups[0]["lr"]
                        avg_loss = running_loss / max(1, running_steps)
                        avg_bce = running_bce / max(1, running_steps)
                        avg_brier = running_brier / max(1, running_steps)
                        avg_kl = running_kl / max(1, running_steps) if teacher is not None else 0.0
                        avg_feat = running_feat / max(1, running_steps) if teacher is not None else 0.0
                        distill_w = args.distill_weight
                        if args.distill_warmup_samples > 0:
                            warmup_frac = min(1.0, global_samples_processed / args.distill_warmup_samples)
                            if args.distill_warmup_mode == "cosine":
                                warmup_frac = 0.5 * (1.0 - math.cos(math.pi * warmup_frac))
                            distill_w = distill_w * warmup_frac
                        # SN34-focused logging
                        log_parts = [f"[train] step={step} lr={lr_now:.2e} loss={avg_loss:.4f}"]
                        log_parts.append(f"stage={stage_name}")
                        log_parts.append(f"brier={avg_brier:.4f}")  # Primary SN34 metric
                        if args.bce_weight > 0:
                            log_parts.append(f"bce={avg_bce:.4f}")
                        if teacher is not None:
                            if distill_kind == "kl":
                                log_parts.append(f"kl={avg_kl:.4f}")
                            else:
                                log_parts.append(f"feat={avg_feat:.4f}")
                            log_parts.append(f"distill_w={distill_w:.3f}")
                        if total_samples_seen > 0:
                            dummy_ratio = dummy_samples_seen / total_samples_seen
                            log_parts.append(f"dummy={dummy_ratio:.3%}")
                        log_parts.append(f"samples={global_samples_processed:,}")
                        print(" ".join(log_parts))
                        running_loss = 0.0
                        running_bce = 0.0
                        running_brier = 0.0
                        running_kl = 0.0
                        running_feat = 0.0
                        running_steps = 0

                    if total_samples_seen >= args.dummy_ratio_warmup_samples and total_samples_seen > 0:
                        dummy_ratio = dummy_samples_seen / total_samples_seen
                        if dummy_ratio > args.max_dummy_ratio:
                            raise RuntimeError(
                                f"Dummy sample ratio too high: {dummy_ratio:.3%} "
                                f"(threshold={args.max_dummy_ratio:.3%}, "
                                f"samples={total_samples_seen}, dummy={dummy_samples_seen}). "
                                "Check stream/download paths and dataset integrity."
                            )

                    do_eval = args.eval_every_steps > 0 and (step % args.eval_every_steps == 0)
                    if do_eval and world_size > 1:
                        dist.barrier()
                    if do_eval:
                        try:
                            acc, bce, brier, mcc, sn34, eval_info = evaluate(
                                model.module if world_size > 1 else model,
                                val_loader,
                                device,
                                mean,
                                std,
                                inference_temp=args.inference_temp,
                                sn34_alpha=args.sn34_alpha,
                                sn34_beta=args.sn34_beta,
                                use_bf16=use_bf16,
                                world_size=world_size,
                                rank=rank,
                                is_main=is_main,
                                macro_avg=args.macro_avg_eval,
                                dataset_idx_to_name=dataset_idx_to_name,
                            )
                            if is_main:
                                print(
                                    f"[eval] step={step} acc={acc:.4f} bce={bce:.4f} "
                                    f"brier={brier:.4f} mcc={mcc:.4f} sn34={sn34:.4f}"
                                )
                                _track_dataset_trends(eval_info, step, is_main)
                            can_save_best = (sn34 > best_sn34) and (mcc >= args.checkpoint_mcc_floor)
                            if can_save_best and args.min_dataset_sn34 > 0 and eval_info["per_dataset"]:
                                min_ds = eval_info["min_sn34"]
                                if min_ds < args.min_dataset_sn34:
                                    can_save_best = False
                                    if is_main:
                                        print(
                                            f"[checkpoint] REJECTED despite macro SN34={sn34:.4f}: "
                                            f"dataset '{eval_info['min_sn34_dataset']}' "
                                            f"SN34={min_ds:.4f} < floor {args.min_dataset_sn34:.4f}"
                                        )
                            if can_save_best:
                                best_sn34 = sn34
                                if is_main:
                                    try:
                                        model_state = model.module.state_dict() if world_size > 1 else model.state_dict()
                                        ckpt = {
                                            "model_state_dict": model_state,
                                            "optimizer_state_dict": optimizer.state_dict(),
                                            "projector_state_dict": proj.state_dict() if proj is not None else None,
                                            "step": step,
                                            "swa_count": swa_count,
                                            "best_sn34": sn34,
                                            "best_brier": brier,
                                            "best_mcc": mcc,
                                            "best_bce": bce,
                                            "best_acc": acc,
                                            "model_id": args.model_id,
                                        }
                                        torch.save(ckpt, Path(args.output_dir) / "best.pt")
                                        print(f"[checkpoint] Saved best.pt (step={step}, SN34={sn34:.4f})")
                                    except Exception as save_err:
                                        print(f"[ERROR] Checkpoint save failed: {save_err}")
                                        print(f"[ERROR] Continuing training without saving...")
                            elif is_main and sn34 > best_sn34 and mcc < args.checkpoint_mcc_floor:
                                print(
                                    f"[checkpoint] Skip best.pt despite SN34 improve: "
                                    f"MCC {mcc:.4f} < floor {args.checkpoint_mcc_floor:.4f}"
                                )
                        except Exception as e:
                            print(f"[ERROR] Evaluation failed at step {step}: {e}")
                            print(f"[ERROR] Skipping evaluation, continuing training...")
                    if do_eval and world_size > 1:
                        dist.barrier()
                    if do_eval:
                        (model.module if world_size > 1 else model).train()

                # SWA collection - all ranks must synchronize
                do_swa = args.swa_start_samples > 0 and global_samples_processed >= next_swa
                if do_swa:
                    if world_size > 1:
                        dist.barrier()  # Sync before SWA check
                    
                    # Run evaluate on all ranks for distributed speed (when quality check needed)
                    cur_sn34 = 0.0
                    if args.swa_min_sn34 > 0:
                        _, _, _, _, cur_sn34, _ = evaluate(
                            model.module if world_size > 1 else model,
                            val_loader, device, mean, std,
                            inference_temp=args.inference_temp,
                            sn34_alpha=args.sn34_alpha, sn34_beta=args.sn34_beta,
                            use_bf16=use_bf16,
                            world_size=world_size,
                            rank=rank,
                            is_main=is_main,
                            macro_avg=args.macro_avg_eval,
                            dataset_idx_to_name=dataset_idx_to_name,
                        )
                        (model.module if world_size > 1 else model).train()
                    
                    if is_main:
                        try:
                            if args.swa_max > 0 and swa_count >= args.swa_max:
                                pass
                            else:
                                swa_ok = cur_sn34 >= args.swa_min_sn34 if args.swa_min_sn34 > 0 else True
                                if args.swa_min_sn34 > 0 and not swa_ok:
                                    print(f"[SWA] skipped @ {global_samples_processed} (SN34={cur_sn34:.4f} < {args.swa_min_sn34})")
                                if swa_ok:
                                    model_state = model.module.state_dict() if world_size > 1 else model.state_dict()
                                    current = {k: v.float().cpu() for k, v in model_state.items()}
                                    if swa_state is None:
                                        swa_state = copy.deepcopy(current)
                                        swa_count = 1
                                    else:
                                        for k in swa_state:
                                            swa_state[k] = (swa_state[k] * swa_count + current[k]) / (swa_count + 1)
                                        swa_count += 1
                                    print(f"[SWA] collected {swa_count} @ {global_samples_processed} samples")
                        except Exception as e:
                            print(f"[ERROR] SWA collection failed: {e}")

                    next_swa += args.swa_every

                    if world_size > 1:
                        dist.barrier()

            if args.stream:
                try:
                    data_manager.cleanup_old_steps(global_step)
                except Exception as e:
                    if is_main:
                        print(f"[ERROR] Cleanup failed: {e}")

        if world_size > 1:
            dist.barrier()
        try:
            acc, bce, brier, mcc, sn34, eval_info = evaluate(
                model.module if world_size > 1 else model,
                val_loader,
                device,
                mean,
                std,
                inference_temp=args.inference_temp,
                sn34_alpha=args.sn34_alpha,
                sn34_beta=args.sn34_beta,
                use_bf16=use_bf16,
                world_size=world_size,
                rank=rank,
                is_main=is_main,
                macro_avg=args.macro_avg_eval,
                dataset_idx_to_name=dataset_idx_to_name,
            )
            if is_main:
                _track_dataset_trends(eval_info, step, is_main)
            can_save_best = (sn34 > best_sn34) and (mcc >= args.checkpoint_mcc_floor)
            if can_save_best and args.min_dataset_sn34 > 0 and eval_info["per_dataset"]:
                min_ds = eval_info["min_sn34"]
                if min_ds < args.min_dataset_sn34:
                    can_save_best = False
                    if is_main:
                        print(
                            f"[checkpoint] REJECTED despite macro SN34={sn34:.4f}: "
                            f"dataset '{eval_info['min_sn34_dataset']}' "
                            f"SN34={min_ds:.4f} < floor {args.min_dataset_sn34:.4f}"
                        )
            if can_save_best:
                best_sn34 = sn34
                if is_main:
                    try:
                        model_state = model.module.state_dict() if world_size > 1 else model.state_dict()
                        ckpt = {
                            "model_state_dict": model_state,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "projector_state_dict": proj.state_dict() if proj is not None else None,
                            "step": step,
                            "swa_count": swa_count,
                            "best_sn34": sn34,
                            "best_brier": brier,
                            "best_mcc": mcc,
                            "best_bce": bce,
                            "best_acc": acc,
                            "model_id": args.model_id,
                        }
                        torch.save(ckpt, Path(args.output_dir) / "best.pt")
                    except Exception as save_err:
                        if is_main:
                            print(f"[ERROR] Epoch-end checkpoint save failed: {save_err}")
            elif is_main and sn34 > best_sn34 and mcc < args.checkpoint_mcc_floor:
                print(
                    f"[checkpoint] Skip epoch best.pt despite SN34 improve: "
                    f"MCC {mcc:.4f} < floor {args.checkpoint_mcc_floor:.4f}"
                )
        except Exception as e:
            if is_main:
                print(f"[ERROR] Epoch-end evaluation failed: {e}")
                print(f"[ERROR] Continuing to next epoch...")
        if world_size > 1:
            dist.barrier()

    # Broadcast swa_state availability to all ranks so they enter the block together
    has_swa = torch.tensor([1 if swa_state is not None else 0], device=device)
    if world_size > 1:
        dist.broadcast(has_swa, src=0)

    if has_swa.item() > 0:
        if world_size > 1:
            dist.barrier()
        try:
            if args.swa_apply:
                # Broadcast SWA weights from rank 0 to all ranks before evaluation
                m = model.module if world_size > 1 else model
                if is_main and swa_state is not None:
                    m.load_state_dict(swa_state, strict=False)
                if world_size > 1:
                    for param in m.parameters():
                        dist.broadcast(param.data, src=0)
                swa_acc, swa_bce, swa_brier, swa_mcc, swa_sn34, _ = evaluate(
                    m, val_loader, device, mean, std, inference_temp=args.inference_temp,
                    sn34_alpha=args.sn34_alpha, sn34_beta=args.sn34_beta, use_bf16=use_bf16,
                    world_size=world_size, rank=rank, is_main=is_main,
                    macro_avg=args.macro_avg_eval, dataset_idx_to_name=dataset_idx_to_name,
                )
            if is_main:
                swa_path = Path(args.output_dir) / "swa.pt"
                torch.save({"model_state_dict": swa_state, "swa_count": swa_count}, swa_path)
                if args.swa_apply:
                    torch.save(
                        {
                            "model_state_dict": swa_state,
                            "best_sn34": swa_sn34,
                            "best_brier": swa_brier,
                            "best_mcc": swa_mcc,
                        },
                        Path(args.output_dir) / "swa_best.pt",
                    )
                print(f"[SWA] saved swa.pt (count={swa_count})")
        except Exception as e:
            if is_main:
                print(f"[ERROR] SWA save failed: {e}")

    # Temperature calibration - run evaluate on all ranks for distributed speed
    try:
        calib_model = model.module if world_size > 1 else model
        best_ckpt_path = Path(args.output_dir) / "best.pt"
        if best_ckpt_path.exists():
            if is_main:
                print(f"\n[Temperature sweep] Loading best checkpoint from {best_ckpt_path}")
            best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            calib_model.load_state_dict(best_ckpt["model_state_dict"])
            if is_main:
                print(f"[Temperature sweep] Loaded best.pt (SN34={best_ckpt.get('best_sn34', 'N/A')})")
        else:
            if is_main:
                print(f"\n[Temperature sweep] No best.pt found, using current model weights")
        if world_size > 1:
            dist.barrier()  # Sync after loading
        
        temps = [float(t) for t in args.calib_temps.split(",") if t.strip()]
        best_t = args.inference_temp
        calib_best_sn34 = 0.0
        best_brier = float("inf")
        if is_main:
            print(f"[Temperature sweep] Testing {len(temps)} temperatures...")
        for t in temps:
            _, bce, brier, mcc, sn34, _ = evaluate(
                calib_model,
                calib_loader,
                device,
                mean,
                std,
                inference_temp=t,
                sn34_alpha=args.sn34_alpha,
                sn34_beta=args.sn34_beta,
                use_bf16=use_bf16,
                world_size=world_size,
                rank=rank,
                is_main=is_main,
                macro_avg=args.macro_avg_eval,
                dataset_idx_to_name=dataset_idx_to_name,
            )
            if is_main:
                print(f"  T={t:.2f}: brier={brier:.4f} mcc={mcc:.4f} sn34={sn34:.4f}")
            if args.calib_metric == "brier":
                if brier < best_brier:
                    best_brier = brier
                    best_t = t
                    calib_best_sn34 = sn34
            else:
                if sn34 > calib_best_sn34:
                    calib_best_sn34 = sn34
                    best_t = t
        if is_main:
            metric_label = "Brier" if args.calib_metric == "brier" else "SN34"
            metric_value = best_brier if args.calib_metric == "brier" else calib_best_sn34
            print(f"[Temperature sweep] Best T={best_t:.2f} ({metric_label}={metric_value:.4f})")
            temp_path = Path(args.output_dir) / "best_temperature.txt"
            temp_path.write_text(f"{best_t}\n")
    except Exception as e:
        if is_main:
            print(f"[ERROR] Temperature calibration failed: {e}")
            print(f"[ERROR] Skipping calibration...")

    if is_main:
        print(f"\n{'='*60}")
        print("✅ TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best SN34: {best_sn34:.4f}")
        print(f"Checkpoints saved to: {args.output_dir}")
        print(f"{'='*60}\n")
    
    cleanup_ddp()


if __name__ == "__main__":
    main()
