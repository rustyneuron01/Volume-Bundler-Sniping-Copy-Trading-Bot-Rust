#!/usr/bin/env python3
"""
InternVideo2 student training wrapper on top of train_videomae.py.

Goals:
- Reuse the full train_videomae pipeline (data, eval, SN34 metrics, SWA, calibration).
- Swap only student backbone to InternVideo2-Stage2_1B style model.
- Provide strategy defaults (hybrid/generalization/sn34) without overriding user args.
- Auto-pick prepared-data CSV defaults when not explicitly provided.
"""

import argparse
import importlib
import os
import sys
import types
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

import train_videomae as base


def _parse_model_spec(model_id: str) -> Tuple[str, str, str]:
    """
    Parse model spec into (mode, repo_or_path, entrypoint).
    Supported forms:
      - hub:<repo>:<entrypoint>
      - local:/path/to/checkpoint.pt
      - bare entrypoint -> assumes OpenGVLab/InternVideo repo
    """
    if model_id.startswith("hub:"):
        payload = model_id[len("hub:") :]
        parts = payload.split(":")
        if len(parts) != 2:
            raise ValueError(
                "Invalid hub model-id format. "
                "Expected hub:<repo>:<entrypoint>, e.g. "
                "hub:OpenGVLab/InternVideo:InternVideo2-Stage2_1B-224p-f4"
            )
        return "hub", parts[0], parts[1]
    if model_id.startswith("local:"):
        ckpt = model_id[len("local:") :]
        if not ckpt:
            raise ValueError("Invalid local model-id format. Expected local:/path/to/checkpoint.pt")
        return "local", ckpt, ""
    return "hub", "OpenGVLab/InternVideo", model_id


def _extract_feature(out: object) -> torch.Tensor:
    """
    Extract pooled [B, D] feature from InternVideo outputs.
    InternVideo2 returns (z, x) or (z, x, attn), where x is clip-level feature.
    """
    if isinstance(out, (list, tuple)):
        if len(out) >= 2 and torch.is_tensor(out[1]):
            feat = out[1]  # x: final pooled clip feature [B, D]
        else:
            raise RuntimeError("InternVideo tuple output missing pooled feature at index 1")
    elif torch.is_tensor(out):
        feat = out
    elif hasattr(out, "last_hidden_state"):
        hidden = out.last_hidden_state
        feat = hidden[:, 0] if hidden.dim() == 3 and hidden.size(1) >= 1 else hidden.mean(dim=1)
    else:
        raise RuntimeError(f"Unsupported output type from backbone: {type(out)}")

    if feat.dim() > 2:
        feat = feat.flatten(start_dim=1)
    return feat


def _install_flash_attn_fallback_stubs() -> None:
    """Provide minimal flash_attn stubs for local InternVideo imports."""
    if "flash_attn.modules.mlp" not in sys.modules:
        mlp_mod = types.ModuleType("flash_attn.modules.mlp")

        class FusedMLP(nn.Module):
            def __init__(self, in_features, hidden_features, activation=None, bias1=True, bias2=True, **kwargs):
                super().__init__()
                self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1)
                self.act = nn.GELU()
                self.fc2 = nn.Linear(hidden_features, in_features, bias=bias2)

            def forward(self, x):
                return self.fc2(self.act(self.fc1(x)))

        mlp_mod.FusedMLP = FusedMLP
        sys.modules["flash_attn.modules.mlp"] = mlp_mod

    if "flash_attn.ops.rms_norm" not in sys.modules:
        rms_mod = types.ModuleType("flash_attn.ops.rms_norm")

        class DropoutAddRMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-6, prenorm=True):
                super().__init__()
                self.norm = nn.LayerNorm(dim, eps=eps)

            def forward(self, x, residual=None, prenorm=True):
                if residual is None:
                    residual = x
                return self.norm(x), residual

        rms_mod.DropoutAddRMSNorm = DropoutAddRMSNorm
        sys.modules["flash_attn.ops.rms_norm"] = rms_mod

    if "flash_attn.flash_attn_interface" not in sys.modules:
        fai = types.ModuleType("flash_attn.flash_attn_interface")

        def _not_available(*args, **kwargs):
            raise RuntimeError("flash_attn kernels unavailable in fallback mode")

        fai.flash_attn_varlen_qkvpacked_func = _not_available
        sys.modules["flash_attn.flash_attn_interface"] = fai

    if "flash_attn.bert_padding" not in sys.modules:
        bp = types.ModuleType("flash_attn.bert_padding")

        def _na(*args, **kwargs):
            raise RuntimeError("flash_attn padding helpers unavailable in fallback mode")

        bp.unpad_input = _na
        bp.pad_input = _na
        sys.modules["flash_attn.bert_padding"] = bp


def _interpolate_stage2_pos_embed(module_sd: dict, model) -> None:
    """Interpolate stage2 pos_embed if temporal grid differs."""
    key = "vision_encoder.pos_embed"
    if key not in module_sd:
        return
    src = module_sd[key]
    dst = model.pos_embed
    if src.shape == dst.shape:
        return
    if src.ndim != 3 or dst.ndim != 3:
        return

    cls_src = src[:, :1, :]
    tok_src = src[:, 1:, :]
    tok_dst = dst[:, 1:, :]
    c = tok_src.shape[-1]

    t_src = 4  # Stage2 checkpoint default temporal grid.
    if tok_src.shape[1] % t_src != 0:
        return
    hw_src = tok_src.shape[1] // t_src
    s_src = int(hw_src**0.5)

    t_dst = int(getattr(model, "T", 8))
    if tok_dst.shape[1] % max(t_dst, 1) != 0:
        return
    hw_dst = tok_dst.shape[1] // max(t_dst, 1)
    s_dst = int(hw_dst**0.5)

    tok_src = tok_src.reshape(1, t_src, s_src, s_src, c).permute(0, 4, 1, 2, 3)
    tok_src = torch.nn.functional.interpolate(
        tok_src, size=(t_dst, s_dst, s_dst), mode="trilinear", align_corners=False
    )
    tok_src = tok_src.permute(0, 2, 3, 4, 1).reshape(1, t_dst * s_dst * s_dst, c)
    module_sd[key] = torch.cat([cls_src, tok_src], dim=1)


class InternVideoBackboneAdapter(nn.Module):
    """
    Adapter so train_videomae can call core.videomae(pixel_values=...) consistently.
    Input: [B, C, T, H, W], Output: pooled [B, D].
    """

    def __init__(self, backbone: nn.Module, runtime_frames: int = 8, expected_input_frames: int = 16):
        super().__init__()
        self.backbone = backbone
        self.runtime_frames = int(runtime_frames)
        self.expected_input_frames = int(expected_input_frames)

    @staticmethod
    def _temporal_fit(x: torch.Tensor, target_frames: int) -> torch.Tensor:
        """Uniformly sample to target length (or edge-pad if too short)."""
        t = int(x.shape[2])
        if t == target_frames:
            return x
        if t > target_frames:
            idx = torch.linspace(0, t - 1, target_frames, device=x.device).long()
            return x.index_select(2, idx)
        pad = target_frames - t
        return torch.cat([x, x[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2)

    def _frame_adapt(self, x: torch.Tensor) -> torch.Tensor:
        # Validator/GasBench path is 16-frame input; keep this as the contract.
        x = self._temporal_fit(x, self.expected_input_frames)
        # Match backbone temporal grid deterministically (e.g. 16 -> 8).
        return self._temporal_fit(x, self.runtime_frames)

    def forward(self, pixel_values=None, **kwargs):
        x = pixel_values
        if x is None and "x" in kwargs:
            x = kwargs["x"]
        if x is None:
            raise ValueError("Expected pixel_values in InternVideoBackboneAdapter")
        if x.dim() != 5:
            raise ValueError(f"Expected [B,C,T,H,W], got {tuple(x.shape)}")

        x = self._frame_adapt(x)
        out = self.backbone(x)
        return _extract_feature(out)


class InternVideo2ForClassification(nn.Module):
    """
    Drop-in replacement for base.VideoMAEv2ForClassification.
    Keeps the same contract expected by train_videomae.py:
      - .videomae attribute
      - .fc_norm, .classifier
      - forward(pixel_values=...) returning object with .logits
    """

    def __init__(self, model_id: str, num_labels: int = 2):
        super().__init__()
        mode, repo_or_path, entrypoint = _parse_model_spec(model_id)
        print(f"[InternVideo2ForClassification] mode={mode} spec={model_id}")

        runtime_frames = 8
        if mode == "hub":
            print(f"[InternVideo2ForClassification] Loading torch.hub: {repo_or_path}::{entrypoint}")
            intern = torch.hub.load(repo_or_path, entrypoint, pretrained=True, trust_repo=True)
            runtime_frames = int(getattr(intern, "T", 8))
        else:
            code_path = os.path.expanduser(
                os.environ.get("INTERNVIDEO2_CODE_PATH", "~/InternVideo/InternVideo2/single_modality")
            )
            if not os.path.isdir(code_path):
                raise RuntimeError(
                    f"INTERNVIDEO2_CODE_PATH not found: {code_path}. "
                    "Set env INTERNVIDEO2_CODE_PATH to InternVideo2/single_modality."
                )
            models_dir = os.path.join(code_path, "models")
            sys.path.insert(0, code_path)
            _install_flash_attn_fallback_stubs()

            models_pkg = types.ModuleType("models")
            models_pkg.__path__ = [models_dir]
            sys.modules["models"] = models_pkg
            teacher_mod = importlib.import_module("models.internvideo2_teacher")
            InternVideo2 = teacher_mod.InternVideo2

            intern = InternVideo2(
                img_size=224,
                patch_size=14,
                embed_dim=1408,
                depth=40,
                num_heads=16,
                mlp_ratio=48 / 11,
                attn_pool_num_heads=16,
                clip_embed_dim=768,
                clip_norm_type="l2",
                return_attn=False,
                clip_return_layer=1,
                clip_return_interval=1,
                num_frames=8,
                tubelet_size=1,
                use_flash_attn=False,
                use_fused_rmsnorm=False,
                use_fused_mlp=False,
            )
            runtime_frames = int(getattr(intern, "T", 8))

            ckpt_path = os.path.expanduser(repo_or_path)
            print(f"[InternVideo2ForClassification] Loading local checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            module_sd = ckpt.get("module", ckpt.get("model", ckpt))
            _interpolate_stage2_pos_embed(module_sd, intern)
            new_sd = {}
            for key, value in module_sd.items():
                if not key.startswith("vision_encoder."):
                    continue
                if "clip_decoder" in key or "final_clip_decoder" in key:
                    continue
                if "clip_pos_embed" in key or "clip_img_pos_embed" in key or "img_pos_embed" in key:
                    continue
                new_sd[key.replace("vision_encoder.", "")] = value
            msg = intern.load_state_dict(new_sd, strict=False)
            print(f"[InternVideo2ForClassification] checkpoint load message: {msg}")

        self.videomae = InternVideoBackboneAdapter(
            intern, runtime_frames=runtime_frames, expected_input_frames=16
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 16, 224, 224, dtype=torch.float32)
            hidden_size = int(_extract_feature(self.videomae(pixel_values=dummy)).shape[-1])
        print(f"[InternVideo2ForClassification] hidden_size={hidden_size}, runtime_frames={runtime_frames}")

        self.fc_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, pixel_values, **kwargs):
        feat = _extract_feature(self.videomae(pixel_values=pixel_values, **kwargs))
        logits = self.classifier(self.fc_norm(feat))

        class Output:
            def __init__(self, logits):
                self.logits = logits

        return Output(logits)


def _has_flag(argv: List[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def _set_default_arg(argv: List[str], flag: str, value: str = "") -> None:
    if not _has_flag(argv, flag):
        if value:
            argv.extend([flag, value])
        else:
            argv.append(flag)


def _pick_existing(paths: List[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _default_intern_model_id() -> str:
    """
    Prefer a local HF-cached Stage2-1B checkpoint for stability.
    Falls back to hub spec when cache is not present.
    """
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--OpenGVLab--InternVideo2-Stage2_1B-224p-f4" / "snapshots"
    if hf_cache.exists():
        candidates = sorted(hf_cache.glob("*/InternVideo2-stage2_1b-224p-f4.pt"))
        if candidates:
            return f"local:{candidates[-1]}"
    return "hub:OpenGVLab/InternVideo:InternVideo2-Stage2_1B-224p-f4"


def _inject_defaults(argv: List[str], strategy: str, root: Path, test_training: bool) -> List[str]:
    prepared = root / "prepared-data"

    # Dataset defaults (only when user didn't pass explicit csv args).
    train_default = _pick_existing(
        [
            prepared / "rebalanced_finetune.csv",
            prepared / "rebalanced_full.csv",
            prepared / "balanced_3.3M_seed789.csv",
            prepared / "master_all.csv",
        ]
    )
    val_default = _pick_existing([prepared / "val_select.csv"])
    calib_default = _pick_existing([prepared / "val_calib.csv"])

    if train_default is not None:
        _set_default_arg(argv, "--train-csv", str(train_default))
    if val_default is not None:
        _set_default_arg(argv, "--val-csv", str(val_default))
    if calib_default is not None:
        _set_default_arg(argv, "--calib-csv", str(calib_default))

    # InternVideo1B default base model (prefer local cache over hub).
    _set_default_arg(argv, "--model-id", _default_intern_model_id())
    _set_default_arg(argv, "--teacher-model-id", "")
    _set_default_arg(argv, "--teacher-frames", "16")
    _set_default_arg(argv, "--calib-metric", "sn34")
    _set_default_arg(argv, "--label-smoothing", "0.0")
    # SWA: default threshold was 2.7M which never triggers for <1M sample datasets.
    # Set to 50% of a typical training run so SWA actually collects snapshots.
    _set_default_arg(argv, "--swa-start-samples", "400000")
    _set_default_arg(argv, "--swa-every", "50000")
    _set_default_arg(argv, "--swa-max", "12")
    _set_default_arg(argv, "--swa-apply")
    # Wider temperature search range for better calibration
    _set_default_arg(argv, "--calib-temps", "0.5,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5,1.7,2.0,2.5,3.0")

    # Strategy presets (user-provided args always win).
    if strategy == "generalization":
        _set_default_arg(argv, "--epochs", "2")
        _set_default_arg(argv, "--lr", "4e-5")
        _set_default_arg(argv, "--backbone-lr-scale", "0.03")
        _set_default_arg(argv, "--layer-lr-decay", "0.85")
        _set_default_arg(argv, "--bce-weight", "0.15")
        _set_default_arg(argv, "--brier-weight", "0.5")
        _set_default_arg(argv, "--distill-weight", "0.0")
        _set_default_arg(argv, "--temporal-skip-prob", "0.05")
        _set_default_arg(argv, "--hflip-prob", "0.2")
        _set_default_arg(argv, "--color-jitter", "0.05")
        _set_default_arg(argv, "--color-jitter-prob", "0.15")
    elif strategy == "sn34":
        _set_default_arg(argv, "--epochs", "1.5")
        _set_default_arg(argv, "--lr", "3e-5")
        _set_default_arg(argv, "--backbone-lr-scale", "0.01")
        _set_default_arg(argv, "--layer-lr-decay", "0.9")
        _set_default_arg(argv, "--bce-weight", "0.05")
        _set_default_arg(argv, "--brier-weight", "0.6")
        _set_default_arg(argv, "--distill-weight", "0.0")
        _set_default_arg(argv, "--temporal-skip-prob", "0.0")
        _set_default_arg(argv, "--hflip-prob", "0.0")
        _set_default_arg(argv, "--color-jitter", "0.0")
        _set_default_arg(argv, "--color-jitter-prob", "0.0")
    else:
        # Hybrid: two-phase training for robust generalization.
        #
        # Phase 1 (first 15%): backbone FROZEN, only head trains.
        #   → Head learns a good classifier on pretrained features.
        #   → No risk of destroying InternVideo2's general video understanding.
        #
        # Phase 2 (15%→25%): backbone gradually unfreezes (linear ramp).
        #   → Backbone adapts to learn deepfake artifacts while head is stable.
        #
        # Phase 3 (25%→100%): full fine-tuning at target LR.
        #   → backbone-lr-scale=0.05 is 20x lower than head — enough to adapt
        #     features but not so aggressive it destroys them.
        #
        # Previous backbone-lr-scale=0.02 made the model a frozen linear probe
        # that memorized dataset identity instead of learning real vs synthetic.
        _set_default_arg(argv, "--epochs", "3")
        _set_default_arg(argv, "--lr", "3e-5")
        _set_default_arg(argv, "--backbone-lr-scale", "0.05")
        _set_default_arg(argv, "--layer-lr-decay", "0.92")
        _set_default_arg(argv, "--backbone-warmup-frac", "0.15")
        _set_default_arg(argv, "--bce-weight", "0.15")
        _set_default_arg(argv, "--brier-weight", "0.45")
        _set_default_arg(argv, "--distill-weight", "0.0")
        _set_default_arg(argv, "--temporal-skip-prob", "0.1")
        _set_default_arg(argv, "--hflip-prob", "0.3")
        _set_default_arg(argv, "--color-jitter", "0.15")
        _set_default_arg(argv, "--color-jitter-prob", "0.3")
        # Dataset balance: weighted sampling so each dataset contributes equally
        # to gradients regardless of size (prevents large datasets from dominating).
        _set_default_arg(argv, "--dataset-balance")
        # Macro-avg eval: checkpoint selected by equal-weight average across all
        # datasets, so the model can't ignore small but critical datasets.
        _set_default_arg(argv, "--macro-avg-eval")

    if test_training:
        _set_default_arg(argv, "--smoke-steps", "3")
        _set_default_arg(argv, "--eval-every-steps", "1")
        _set_default_arg(argv, "--log-every", "1")
        _set_default_arg(argv, "--swa-start-samples", "0")
        _set_default_arg(argv, "--output-dir", str(root / "checkpoints" / "internvideo_test"))

    return argv


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--strategy",
        type=str,
        default="hybrid",
        choices=["hybrid", "generalization", "sn34"],
        help="Training preset. hybrid is recommended for stable holdout performance.",
    )
    parser.add_argument(
        "--test-training",
        action="store_true",
        help="Run very short smoke training while keeping full pipeline path.",
    )

    wrapper_args, passthrough = parser.parse_known_args(sys.argv[1:])

    root = Path(__file__).resolve().parent
    final_args = _inject_defaults(passthrough, wrapper_args.strategy, root, wrapper_args.test_training)

    print(f"[train_intern] strategy={wrapper_args.strategy}")
    if wrapper_args.test_training:
        print("[train_intern] --test-training enabled")

    # Monkeypatch only student model class and preserve base pipeline behavior.
    base.VideoMAEv2ForClassification = InternVideo2ForClassification

    sys.argv = [sys.argv[0]] + final_args
    base.main()


if __name__ == "__main__":
    main()
