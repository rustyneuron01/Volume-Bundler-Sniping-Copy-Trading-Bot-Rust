#!/usr/bin/env python3
"""
MViTv2-L training wrapper on top of train_videomae.py.

Architecture:
  MViTv2-L (timm, 218M backbone params, ImageNet-21k pretrained)
  → per-frame feature extraction [B*T, 3, 224, 224] → [B*T, 1152]
  → TemporalTransformer [B, T, 1152] → [B, 512]
  → fc_norm + classifier → [B, 2] logits

MViTv2-L vs InternVideo2 — why the strategy differs:
  InternVideo2 is a VIDEO model: it already understands temporal patterns.
  Fine-tuning only teaches it "which temporal patterns = fake vs real."
  So InternVideo2 needs minimal backbone adaptation and Brier-focused loss
  (it already classifies well, just needs better calibration).

  MViTv2-L is an IMAGE model: it processes each frame independently.
  It must learn TWO things from scratch:
    1. Adapt spatial features for forensic artifact detection
    2. Learn temporal aggregation across frames (via TemporalTransformer)
  So MViTv2-L needs stronger backbone adaptation, CE-focused loss
  (it needs to LEARN correct classification first), and the temporal
  transformer must train at full LR since it's randomly initialized.

  MViTv2-L's advantage: its multi-scale pooled attention naturally
  captures artifacts at all spatial scales simultaneously — fine textures
  (stage 1, 56×56), local patterns (stage 2, 28×28), mid-level
  structures (stage 3, 14×14), and global context (stage 4, 7×7).
  This is ideal for detecting the multi-scale artifacts AI generators produce.

Training Strategy (based on DeMamba/GenVideo, Community Forensics CVPR 2025,
  GenD CVPR 2025, DeepfakeBench NeurIPS 2023):

  SMALL DATASET HANDLING (training):
    1. Inverse-frequency weighted sampling (compute_dataset_weights):
       each dataset gets gradient contribution ∝ 1/N_datasets regardless
       of its sample count, with real/fake balanced within each dataset.
    2. max_repeat=20 cap: prevents memorization of tiny datasets while
       still oversampling them meaningfully (20x per epoch max).
    3. Stratified shuffle: every step block sees samples from ALL datasets,
       not just large ones.  Prevents "starvation" across training steps.

  SMALL DATASET HANDLING (evaluation + checkpoint selection):
    4. Macro-averaged SN34: each dataset contributes equally to the metric
       that drives checkpoint selection (same approach as Community Forensics'
       evalSeparately + meanAP/meanAcc/meanLoss).
    5. Per-dataset SN34 floor (--min-dataset-sn34): REJECTS checkpoints
       where ANY dataset's SN34 falls below a minimum threshold, even if
       the macro average improved.  Prevents "sacrificing" small datasets.
    6. Per-dataset trend tracking: monitors each dataset's SN34 across
       evaluation rounds and WARNS when any dataset regresses by >0.10
       from its previous best (detects catastrophic forgetting per-source).
    7. Confidence distribution: tracks % uncertain (0.4-0.6) vs confident
       predictions to diagnose calibration quality for Brier optimization.

  HOW TO VERIFY TRAINING IS GOOD (what to monitor):
    - macro_sn34 should steadily increase (primary metric)
    - min_sn34 across datasets should NOT decrease (no dataset sacrifice)
    - uncertain_pct should decrease over training (model gaining confidence)
    - No REGRESSION WARNINGs in output (no forgetting of learned datasets)
    - Worst-10 datasets printout: their SN34 should improve, not stagnate
    - Temperature sweep should find T near 1.0 (well-calibrated logits)

Usage:
  # Single GPU (use rebalanced_full.csv, NOT gasbench_balanced.csv which has 97% val leakage)
  python train_mvitv2.py --train-csv prepared-data/rebalanced_full.csv \\
      --val-csv prepared-data/val_select.csv

  # 4x H200 DDP
  torchrun --nproc_per_node=4 train_mvitv2.py --strategy hybrid --h200-4x --stream

  # Quick test
  python train_mvitv2.py --test-training
"""

import argparse
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import timm

import train_videomae as base


MVITV2_TIMM_NAME = "mvitv2_large_cls.fb_inw21k"
MVITV2_FEATURE_DIM = 1152
TEMPORAL_DIM = 512
TEMPORAL_HEADS = 8
TEMPORAL_LAYERS = 2


class TemporalTransformer(nn.Module):
    """Cross-frame temporal aggregation via CLS-token Transformer.

    Learns temporal patterns that distinguish real from AI-generated video:
    frame-to-frame consistency, motion naturalness, temporal artifact
    patterns (flickering, discontinuities).

    This module is randomly initialized and must train at full LR.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, num_heads: int = 8,
                 num_layers: int = 2, max_frames: int = 16, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → [B, hidden_dim]"""
        B, T, _ = x.shape
        x = self.proj(x)
        x = x + self.pos_embed[:, :T, :]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return self.norm(x[:, 0])


class MViTv2BackboneAdapter(nn.Module):
    """
    Wraps timm MViTv2-L image backbone + temporal transformer.

    The training loop (train_videomae.py) calls core.videomae(pixel_values=...)
    and expects [B, D] output (2D). If we returned [B, T, D] (3D), the loop
    would mean-pool it, bypassing the temporal transformer. So the temporal
    transformer MUST run inside this adapter to produce 2D output.

    forward(pixel_values=[B,C,T,H,W]) → [B, temporal_dim]
    """

    def __init__(self, temporal: TemporalTransformer, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            MVITV2_TIMM_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=True)
            print("[MViTv2] Gradient checkpointing enabled")

        # The temporal module is SHARED with the parent MViTv2ForClassification.
        # Parameters are owned by parent's classifier_temporal (→ head LR group),
        # not by this adapter (→ backbone LR group).
        self.temporal = temporal

    def forward(self, pixel_values=None, **kwargs):
        x = pixel_values
        if x is None:
            x = kwargs.get("x")
        if x is None:
            raise ValueError("Expected pixel_values in MViTv2BackboneAdapter")
        if x.dim() != 5:
            raise ValueError(f"Expected [B,C,T,H,W], got {tuple(x.shape)}")

        B, C, T, H, W = x.shape
        # [B,C,T,H,W] → [B*T, C, H, W] for per-frame backbone
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        features = self.backbone(x)  # [B*T, 1152]
        features = features.reshape(B, T, -1)  # [B, T, 1152]
        return self.temporal(features)  # [B, 512]


class MViTv2ForClassification(nn.Module):
    """
    Drop-in replacement for base.VideoMAEv2ForClassification.

    ARCHITECTURE:
      .classifier_temporal (TemporalTransformer)
          → params named classifier_temporal.* → HEAD LR group (full LR)
      .videomae (MViTv2BackboneAdapter)
          .backbone (timm mvitv2_large_cls) → BACKBONE LR group (lower LR)
          .temporal (shared ref to classifier_temporal, no duplicate params)
      .fc_norm (LayerNorm 512) → HEAD LR group
      .classifier (Linear 512→2) → HEAD LR group

    The base pipeline groups params by name substring:
      'classifier' or 'fc_norm' → head (full LR, never frozen during warmup)
      everything else → backbone_other (backbone_lr_scale × LR)

    classifier_temporal is registered BEFORE videomae so PyTorch's
    named_parameters() lists its params under classifier_temporal.*
    (matching 'classifier' → head group). The shared reference inside
    videomae.temporal doesn't create duplicate params.
    """

    def __init__(self, model_id: str = "", num_labels: int = 2):
        super().__init__()
        print(f"[MViTv2ForClassification] Loading {MVITV2_TIMM_NAME} from timm")

        # Temporal transformer — registered FIRST so its params get
        # classifier_temporal.* naming → head LR group
        self.classifier_temporal = TemporalTransformer(
            input_dim=MVITV2_FEATURE_DIM,
            hidden_dim=TEMPORAL_DIM,
            num_heads=TEMPORAL_HEADS,
            num_layers=TEMPORAL_LAYERS,
        )

        # Backbone adapter — receives shared ref to temporal
        self.videomae = MViTv2BackboneAdapter(
            temporal=self.classifier_temporal,
            pretrained=True,
        )

        hidden_size = TEMPORAL_DIM
        backbone_params = sum(p.numel() for p in self.videomae.backbone.parameters())
        temporal_params = sum(p.numel() for p in self.classifier_temporal.parameters())
        print(f"[MViTv2ForClassification] backbone={backbone_params/1e6:.1f}M, "
              f"temporal={temporal_params/1e6:.1f}M, hidden={hidden_size}")

        self.fc_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, pixel_values, **kwargs):
        feat = self.videomae(pixel_values=pixel_values, **kwargs)  # [B, 512]
        logits = self.classifier(self.fc_norm(feat))

        class Output:
            def __init__(self, logits):
                self.logits = logits
        return Output(logits)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_flag(argv: List[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def _set_default_arg(argv: List[str], flag: str, value: str = "") -> None:
    if not _has_flag(argv, flag):
        if value:
            argv.extend([flag, value])
        else:
            argv.append(flag)


def _pick_existing(paths: List[Path]):
    for p in paths:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Strategy defaults — designed for MViTv2-L (NOT copied from InternVideo2)
# ---------------------------------------------------------------------------

def _inject_defaults(argv: List[str], strategy: str, root: Path, test_training: bool) -> List[str]:
    prepared = root / "prepared-data"

    # rebalanced_full preferred: 903K rows, all 52 datasets, only 0.1%
    # overlap with val_select (866 samples).
    # DO NOT use gasbench_balanced.csv — 97% data leakage with val_select.
    train_default = _pick_existing([
        prepared / "rebalanced_full.csv",
        prepared / "rebalanced_finetune.csv",
        prepared / "balanced_3.3M_seed789.csv",
        prepared / "master_all.csv",
    ])
    val_default = _pick_existing([prepared / "val_select.csv"])
    calib_default = _pick_existing([prepared / "val_calib.csv"])

    if train_default is not None:
        _set_default_arg(argv, "--train-csv", str(train_default))
    if val_default is not None:
        _set_default_arg(argv, "--val-csv", str(val_default))
    if calib_default is not None:
        _set_default_arg(argv, "--calib-csv", str(calib_default))

    _set_default_arg(argv, "--model-id", MVITV2_TIMM_NAME)
    _set_default_arg(argv, "--teacher-model-id", "")
    _set_default_arg(argv, "--calib-metric", "sn34")
    _set_default_arg(argv, "--output-dir", str(root / "checkpoints" / "mvitv2_large"))
    # Wide temperature sweep — MViTv2 is newly trained so calibration matters more
    _set_default_arg(argv, "--calib-temps", "0.5,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5,1.7,2.0,2.5,3.0")

    if strategy == "generalization":
        # Conservative: focus on generalization over peak SN34
        _set_default_arg(argv, "--epochs", "2")
        _set_default_arg(argv, "--lr", "4e-5")
        _set_default_arg(argv, "--backbone-lr-scale", "0.08")
        _set_default_arg(argv, "--backbone-warmup-frac", "0.05")
        _set_default_arg(argv, "--label-smoothing", "0.05")
        _set_default_arg(argv, "--bce-weight", "0.40")
        _set_default_arg(argv, "--brier-weight", "0.30")
        _set_default_arg(argv, "--distill-weight", "0.0")
        _set_default_arg(argv, "--hflip-prob", "0.3")
        _set_default_arg(argv, "--color-jitter", "0.10")
        _set_default_arg(argv, "--color-jitter-prob", "0.25")
        _set_default_arg(argv, "--temporal-skip-prob", "0.05")
        _set_default_arg(argv, "--focal-gamma", "1.5")
        _set_default_arg(argv, "--jpeg-prob", "0.30")
        _set_default_arg(argv, "--jpeg-quality-min", "50")
        _set_default_arg(argv, "--jpeg-quality-max", "100")
        _set_default_arg(argv, "--gaussian-noise-prob", "0.20")
        _set_default_arg(argv, "--gaussian-blur-prob", "0.15")
        _set_default_arg(argv, "--swa-start-samples", "400000")
        _set_default_arg(argv, "--swa-every", "50000")
        _set_default_arg(argv, "--swa-max", "12")
        _set_default_arg(argv, "--swa-apply")
    elif strategy == "sn34":
        # Aggressive: maximize SN34 at potential cost of generalization
        _set_default_arg(argv, "--epochs", "2")
        _set_default_arg(argv, "--lr", "3e-5")
        _set_default_arg(argv, "--backbone-lr-scale", "0.05")
        _set_default_arg(argv, "--backbone-warmup-frac", "0.0")
        _set_default_arg(argv, "--label-smoothing", "0.0")
        _set_default_arg(argv, "--bce-weight", "0.20")
        _set_default_arg(argv, "--brier-weight", "0.50")
        _set_default_arg(argv, "--distill-weight", "0.0")
        _set_default_arg(argv, "--hflip-prob", "0.0")
        _set_default_arg(argv, "--color-jitter", "0.0")
        _set_default_arg(argv, "--color-jitter-prob", "0.0")
        _set_default_arg(argv, "--temporal-skip-prob", "0.0")
        _set_default_arg(argv, "--focal-gamma", "0.0")
        _set_default_arg(argv, "--jpeg-prob", "0.0")
        _set_default_arg(argv, "--gaussian-noise-prob", "0.0")
        _set_default_arg(argv, "--gaussian-blur-prob", "0.0")
        _set_default_arg(argv, "--swa-start-samples", "300000")
        _set_default_arg(argv, "--swa-every", "50000")
        _set_default_arg(argv, "--swa-max", "12")
        _set_default_arg(argv, "--swa-apply")
    else:
        # Hybrid — balanced strategy designed specifically for MViTv2-L
        #
        # KEY DIFFERENCES from InternVideo2 strategy:
        #
        # 1. LOSS: CE-heavy, not Brier-heavy
        #    InternVideo2 already classifies well (video model) → optimize Brier
        #    MViTv2-L needs to LEARN classification (image model) → optimize CE
        #    bce_weight=0.35 vs InternVideo2's 0.15
        #    brier_weight=0.35 vs InternVideo2's 0.45
        #
        # 2. BACKBONE LR: Higher scale (0.10 vs 0.05)
        #    InternVideo2 was already trained on video → minimal adaptation
        #    MViTv2-L is image-pretrained → needs stronger forensic adaptation
        #    Its multi-scale features must adapt to detect artifacts at all scales
        #
        # 3. WARMUP: Shorter freeze (5% vs 15%)
        #    The temporal transformer is randomly initialized and needs to
        #    start learning quickly. 5% is enough for the classifier to get
        #    a basic signal before backbone starts adapting.
        #
        # 4. NO label smoothing
        #    Hard labels produce better Brier scores (exact 0/1 targets).
        #    The model needs sharp decision boundaries for MCC.
        #
        # 5. SPATIAL augmentations over temporal
        #    MViTv2-L is a spatial model, so spatial augmentations (hflip,
        #    color_jitter) matter more. temporal_skip has less effect because
        #    the backbone processes frames independently anyway.
        #
        # 6. ROBUSTNESS AUGMENTATIONS (new, from research):
        #    JPEG compression is the #1 augmentation for generalization to
        #    unknown generators (+4.4% AUROC, Pellegrini et al. 2025).
        #    Gaussian noise/blur force the model to learn genuine artifacts
        #    instead of brittle high-frequency patterns.
        #
        # 7. FOCAL LOSS for per-dataset >95% accuracy:
        #    With gamma=2.0, easy examples get 100x less gradient than hard
        #    ones. Small datasets with hard examples get massive gradient.
        #
        # 8. THREE-STAGE SCHEDULE (inherited from base pipeline):
        #    Stage 1 (0-60%): Generalize — CE-focused, full augmentation
        #    Stage 2 (60-85%): Stabilize — balanced, reduced augmentation
        #    Stage 3 (85-100%): SN34 — Brier-focused, no augmentation
        #
        _set_default_arg(argv, "--epochs", "3")
        _set_default_arg(argv, "--lr", "5e-5")
        _set_default_arg(argv, "--backbone-lr-scale", "0.10")
        _set_default_arg(argv, "--backbone-warmup-frac", "0.05")
        _set_default_arg(argv, "--label-smoothing", "0.0")
        _set_default_arg(argv, "--bce-weight", "0.35")
        _set_default_arg(argv, "--brier-weight", "0.35")
        _set_default_arg(argv, "--distill-weight", "0.0")
        _set_default_arg(argv, "--hflip-prob", "0.3")
        _set_default_arg(argv, "--color-jitter", "0.10")
        _set_default_arg(argv, "--color-jitter-prob", "0.25")
        _set_default_arg(argv, "--temporal-skip-prob", "0.03")
        # Focal loss: gamma=2.0 aggressively focuses on hard examples
        _set_default_arg(argv, "--focal-gamma", "2.0")
        # JPEG compression: most impactful augmentation per Pellegrini 2025
        _set_default_arg(argv, "--jpeg-prob", "0.30")
        _set_default_arg(argv, "--jpeg-quality-min", "40")
        _set_default_arg(argv, "--jpeg-quality-max", "100")
        # Noise + blur: force learning of genuine generator artifacts
        _set_default_arg(argv, "--gaussian-noise-prob", "0.20")
        _set_default_arg(argv, "--gaussian-blur-prob", "0.15")
        _set_default_arg(argv, "--swa-start-samples", "400000")
        _set_default_arg(argv, "--swa-every", "50000")
        _set_default_arg(argv, "--swa-max", "12")
        _set_default_arg(argv, "--swa-apply")

    # Common: dataset balancing + macro eval (critical for all strategies).
    # Research (Community Forensics CVPR 2025, DeMamba/GenVideo) confirms
    # that per-source macro-averaged evaluation is the standard approach
    # in competitive benchmarks.  Without these, large datasets dominate
    # both gradient contribution (training) and checkpoint selection (eval).
    _set_default_arg(argv, "--dataset-balance")
    _set_default_arg(argv, "--macro-avg-eval")
    # Floor guarantee: reject checkpoints where ANY dataset's SN34 drops
    # below this threshold.  Prevents the optimizer from "sacrificing"
    # small datasets to improve the macro average via large ones.
    # 0.15 is deliberately lenient — early training will have low per-dataset
    # SN34, and we don't want to block ALL checkpoints.  As training
    # progresses, the actual min-dataset-SN34 should climb well above this.
    _set_default_arg(argv, "--min-dataset-sn34", "0.15")

    if test_training:
        _set_default_arg(argv, "--smoke-steps", "3")
        _set_default_arg(argv, "--eval-every-steps", "1")
        _set_default_arg(argv, "--log-every", "1")
        _set_default_arg(argv, "--swa-start-samples", "0")
        _set_default_arg(argv, "--output-dir", str(root / "checkpoints" / "mvitv2_test"))

    return argv


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--strategy", type=str, default="hybrid",
        choices=["hybrid", "generalization", "sn34"],
        help="Training preset. hybrid is recommended.",
    )
    parser.add_argument(
        "--test-training", action="store_true",
        help="Short smoke test keeping full pipeline path.",
    )

    wrapper_args, passthrough = parser.parse_known_args(sys.argv[1:])
    root = Path(__file__).resolve().parent
    final_args = _inject_defaults(passthrough, wrapper_args.strategy, root, wrapper_args.test_training)

    print(f"[train_mvitv2] strategy={wrapper_args.strategy}")
    print(f"[train_mvitv2] backbone={MVITV2_TIMM_NAME} (ImageNet-21k, 218M params)")
    print(f"[train_mvitv2] temporal=TemporalTransformer(1152→{TEMPORAL_DIM}, "
          f"{TEMPORAL_HEADS}h, {TEMPORAL_LAYERS}L)")
    print(f"[train_mvitv2] architecture: per-frame MViTv2-L → TemporalTransformer → classifier")
    if wrapper_args.test_training:
        print("[train_mvitv2] --test-training enabled")

    base.VideoMAEv2ForClassification = MViTv2ForClassification

    sys.argv = [sys.argv[0]] + final_args
    base.main()


if __name__ == "__main__":
    main()
