#!/usr/bin/env python3
"""
InternVideo2 student training with full pipeline parity to train_videomae_v2.py.

This script reuses train_videomae_v2.py end-to-end (download/streaming, preprocessing,
evaluation, SWA, temperature sweep, checkpointing), and only swaps the student model.

Extra convenience:
- --test-training: run a short smoke training to validate pipeline before full run.
"""

import os
import sys
from typing import Tuple

import torch
import torch.nn as nn

import train_videomae_v2 as base


def _parse_model_spec(model_id: str) -> Tuple[str, str, str]:
    """
    Parse model spec into (mode, repo_or_path, entrypoint).
    mode:
      - "hub"   -> repo + entrypoint
      - "local" -> checkpoint path + ""
    """
    if model_id.startswith("hub:"):
        payload = model_id[len("hub:") :]
        parts = payload.split(":")
        if len(parts) != 2:
            raise ValueError(
                "Invalid hub model-id format. "
                "Expected: hub:<repo>:<entrypoint>, "
                "e.g. hub:OpenGVLab/InternVideo:InternVideo2-Stage2_1B-224p-f4"
            )
        return "hub", parts[0], parts[1]
    if model_id.startswith("local:"):
        ckpt = model_id[len("local:") :]
        if not ckpt:
            raise ValueError("Invalid local model-id format. Expected: local:/path/to/checkpoint.pt")
        return "local", ckpt, ""
    # Backward-friendly default: treat bare id as hub entrypoint under OpenGVLab/InternVideo
    return "hub", "OpenGVLab/InternVideo", model_id


def _extract_feature(out: object) -> torch.Tensor:
    """Extract [B, D] pooled feature from common InternVideo outputs."""
    if isinstance(out, (list, tuple)):
        if len(out) >= 2 and torch.is_tensor(out[1]):
            feat = out[1]
        elif len(out) >= 1 and torch.is_tensor(out[0]):
            feat = out[0]
        else:
            raise RuntimeError("Unable to parse tuple/list InternVideo output")
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


class InternVideo2ForClassification(nn.Module):
    """
    Drop-in replacement for base.VideoMAEv2ForClassification.
    Keeps the same contract used by train_videomae_v2.py:
      - .videomae attribute for optimizer/group logic
      - forward(pixel_values=...) returning object with .logits
    """

    def __init__(self, model_id: str, num_labels: int = 2):
        super().__init__()
        mode, repo_or_path, entrypoint = _parse_model_spec(model_id)
        print(f"[InternVideo2ForClassification] mode={mode} spec={model_id}")

        if mode == "hub":
            print(f"[InternVideo2ForClassification] Loading torch.hub: {repo_or_path}::{entrypoint}")
            self.videomae = torch.hub.load(
                repo_or_path,
                entrypoint,
                pretrained=True,
                trust_repo=True,
            )
        else:
            code_path = os.path.expanduser(
                os.environ.get("INTERNVIDEO2_CODE_PATH", "~/InternVideo/InternVideo2/single_modality")
            )
            if not os.path.isdir(code_path):
                raise RuntimeError(
                    f"INTERNVIDEO2_CODE_PATH not found: {code_path}. "
                    "Set env INTERNVIDEO2_CODE_PATH to InternVideo2/single_modality."
                )
            sys.path.insert(0, code_path)
            from models.internvideo2_teacher import InternVideo2

            # Keep shape expectations aligned with 16-frame training pipeline.
            self.videomae = InternVideo2(
                img_size=224,
                patch_size=14,
                embed_dim=3200,
                depth=48,
                num_heads=25,
                mlp_ratio=4,
                attn_pool_num_heads=16,
                clip_embed_dim=768,
                clip_norm_type="l2",
                return_attn=True,
                clip_return_layer=1,
                clip_return_interval=1,
                num_frames=16,
                tubelet_size=1,
            )
            ckpt_path = os.path.expanduser(repo_or_path)
            print(f"[InternVideo2ForClassification] Loading local checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "model" in ckpt:
                ckpt = ckpt["model"]
            elif "module" in ckpt:
                ckpt = ckpt["module"]
            msg = self.videomae.load_state_dict(ckpt, strict=False)
            print(f"[InternVideo2ForClassification] checkpoint load message: {msg}")

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 16, 224, 224, dtype=torch.float32)
            out = self.videomae(dummy)
            hidden_size = int(_extract_feature(out).shape[-1])
        print(f"[InternVideo2ForClassification] hidden_size={hidden_size}")

        self.fc_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, pixel_values, **kwargs):
        out = self.videomae(pixel_values)
        feat = _extract_feature(out)
        logits = self.classifier(self.fc_norm(feat))

        class Output:
            def __init__(self, logits):
                self.logits = logits

        return Output(logits)


def _pop_flag(argv, flag: str) -> bool:
    if flag in argv:
        argv.remove(flag)
        return True
    return False


def _inject_default_args():
    """
    Keep base CLI surface but set InternVideo defaults when user omitted them.
    Also supports quick test mode: --test-training.
    """
    argv = sys.argv[1:]

    test_mode = _pop_flag(argv, "--test-training")

    if "--model-id" not in argv:
        argv += ["--model-id", "hub:OpenGVLab/InternVideo:InternVideo2-Stage2_1B-224p-f4"]

    # Keep teacher frame sampling at 16 unless explicitly overridden.
    if "--teacher-frames" not in argv:
        argv += ["--teacher-frames", "16"]

    if test_mode:
        # Keep test deterministic and short, while still exercising the full pipeline.
        if "--smoke-steps" not in argv:
            argv += ["--smoke-steps", "3"]
        if "--eval-every-steps" not in argv:
            argv += ["--eval-every-steps", "1"]
        if "--log-every" not in argv:
            argv += ["--log-every", "1"]
        if "--swa-start-samples" not in argv:
            argv += ["--swa-start-samples", "0"]
        if "--swa-apply" in argv:
            # Allow base parser to ignore this in test path; keep as-is if provided.
            pass
        if "--output-dir" not in argv:
            argv += ["--output-dir", "checkpoints/internvideo_test"]
        print("[train_internvideo] --test-training enabled")
        print("[train_internvideo] Injected: --smoke-steps 3 --eval-every-steps 1 --log-every 1 --swa-start-samples 0")

    sys.argv = [sys.argv[0]] + argv


def main():
    # Monkeypatch only student class; preserve all base pipeline behavior.
    base.VideoMAEv2ForClassification = InternVideo2ForClassification
    _inject_default_args()
    base.main()


if __name__ == "__main__":
    main()
