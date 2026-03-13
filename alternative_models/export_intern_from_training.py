#!/usr/bin/env python3
"""
Export a trained InternVideo discriminator checkpoint (from train_intern/train_videomae)
to GasBench custom PyTorch package format:
  - model_config.yaml
  - model.py
  - model.safetensors
  - <output_dir>.zip
"""

import argparse
from pathlib import Path
import re
import zipfile

import torch
from safetensors.torch import save_file


def build_model_py(temperature: float, input_format: str, inlined_internvideo_source: str) -> str:
    template = '''#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange

__INLINED_INTERNVIDEO_SOURCE__


def _extract_feature(out: object) -> torch.Tensor:
    if isinstance(out, (list, tuple)):
        if len(out) >= 2 and torch.is_tensor(out[1]):
            feat = out[1]  # InternVideo pooled clip feature x
        else:
            raise RuntimeError("InternVideo tuple output missing pooled feature at index 1")
    elif torch.is_tensor(out):
        feat = out
    elif hasattr(out, "last_hidden_state"):
        hidden = out.last_hidden_state
        feat = hidden[:, 0] if hidden.dim() == 3 and hidden.size(1) >= 1 else hidden.mean(dim=1)
    else:
        raise RuntimeError(f"Unsupported output type from backbone: {{type(out)}}")
    if feat.dim() > 2:
        feat = feat.flatten(start_dim=1)
    return feat


class InternVideo2BinaryWrapper(nn.Module):
    def __init__(self, hidden_size=__HIDDEN_SIZE__, temperature=__TEMPERATURE__, input_format="__INPUT_FORMAT__"):
        super().__init__()

        self.backbone = InternVideo2(
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
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)
        self.temperature = float(temperature)
        self.input_format = input_format.upper()
        if self.input_format not in ("RGB", "BGR"):
            raise ValueError(f"input_format must be RGB or BGR, got {{self.input_format}}")
        self.expected_input_frames = 16
        self.runtime_frames = 8
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1, 1))

    @staticmethod
    def _temporal_fit(x: torch.Tensor, target_frames: int) -> torch.Tensor:
        t = int(x.shape[2])
        if t == target_frames:
            return x
        if t > target_frames:
            idx = torch.linspace(0, t - 1, target_frames, device=x.device).long()
            return x.index_select(2, idx)
        pad = target_frames - t
        return torch.cat([x, x[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2)

    def forward(self, x):
        # gasbench validator input: [B, T, C, H, W] uint8 (T expected as 16)
        if x.dim() != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {{tuple(x.shape)}}")
        x = x.float() / 255.0
        if self.input_format == "BGR":
            x = x[:, :, [2, 1, 0], :, :]
        x = x.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W]
        # Match train_videomae/train_intern preprocessing exactly.
        x = (x - self.mean) / self.std
        x = self._temporal_fit(x, self.expected_input_frames)
        x = self._temporal_fit(x, self.runtime_frames)
        feat = _extract_feature(self.backbone(x))
        logits = self.classifier(self.fc_norm(feat))
        return logits / max(self.temperature, 1e-6)


def load_model(weights_path: str, num_classes: int = 2, temperature: float = __TEMPERATURE__, input_format: str = "__INPUT_FORMAT__"):
    from safetensors.torch import load_file

    state_dict = load_file(weights_path)
    hidden_size = int(state_dict["classifier.weight"].shape[1])
    model = InternVideo2BinaryWrapper(hidden_size=hidden_size, temperature=temperature, input_format=input_format)
    msg = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"mean", "std"}
    missing = set(msg.missing_keys)
    unexpected = set(msg.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading export weights: {{sorted(unexpected)}}")
    if not missing.issubset(allowed_missing):
        raise RuntimeError(
            f"Missing required keys while loading export weights: {{sorted(missing - allowed_missing)}} "
            f"(all missing={{sorted(missing)}})"
        )
    print("load_state_dict:", msg)
    model.train(False)
    return model
'''
    return (
        template
        .replace("__INLINED_INTERNVIDEO_SOURCE__", inlined_internvideo_source)
        .replace("__TEMPERATURE__", str(float(temperature)))
        .replace("__INPUT_FORMAT__", input_format)
    )


def convert_checkpoint_to_export_state(model_state_dict: dict) -> dict:
    out = {}
    backbone_count = 0
    for key, value in model_state_dict.items():
        if key.startswith("videomae.backbone."):
            out["backbone." + key[len("videomae.backbone.") :]] = value
            backbone_count += 1
        elif key.startswith("fc_norm.") or key.startswith("classifier."):
            out[key] = value
    if backbone_count == 0:
        raise ValueError(
            "No backbone weights found. Expected keys starting with 'videomae.backbone.'. "
            "Make sure --checkpoint points to train_intern/train_videomae best.pt."
        )
    required = ["classifier.weight", "classifier.bias", "fc_norm.weight", "fc_norm.bias"]
    missing = [k for k in required if k not in out]
    if missing:
        raise ValueError(f"Missing required classifier keys in checkpoint: {missing}")
    return out


def _sanitized_flash_attention_class_py() -> str:
    return '''import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashAttention(nn.Module):
    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None, max_s=None, need_weights=False):
        if need_weights:
            raise NotImplementedError("need_weights=True is not supported in export runtime")
        if cu_seqlens is not None:
            raise NotImplementedError("Packed flash attention path is not supported in export runtime")
        if key_padding_mask is not None:
            raise NotImplementedError("key_padding_mask path is not supported in export runtime")
        if qkv.dim() != 5:
            raise ValueError(f"Expected qkv shape [B,S,3,H,D], got {tuple(qkv.shape)}")
        q, k, v = qkv.unbind(dim=2)  # [B,S,H,D]
        q = q.transpose(1, 2)  # [B,H,S,D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=causal,
        )
        out = out.transpose(1, 2)  # [B,S,H,D]
        return out, None
'''


def _sanitize_internvideo_teacher_source(src_text: str) -> str:
    # Drop blocked/non-allowlisted imports and provide local fallbacks.
    blocked_import_patterns = [
        r"^import os\s*$",
        r"^from collections import OrderedDict\s*$",
        r"^from torch import nn\s*$",
        r"^import torch\.utils\.checkpoint as checkpoint\s*$",
        r"^from flash_attn\.modules\.mlp import FusedMLP\s*$",
        r"^from flash_attn\.ops\.rms_norm import DropoutAddRMSNorm\s*$",
        r"^from \.pos_embed import .*",
        r"^from \.flash_attention_class import FlashAttention\s*$",
        r"^import time\s*$",
        r"^from fvcore\.nn import FlopCountAnalysis\s*$",
        r"^from fvcore\.nn import flop_count_table\s*$",
    ]
    lines = src_text.splitlines()
    kept = []
    for line in lines:
        if any(re.match(pat, line.strip()) for pat in blocked_import_patterns):
            continue
        kept.append(line)
    text = "\n".join(kept)
    text = text.replace("checkpoint.checkpoint(", "torch.utils.checkpoint.checkpoint(")

    # Replace filesystem-dependent checkpoint defaults with inert literals.
    text = re.sub(
        r"MODEL_PATH\s*=.*?\n_MODELS\s*=\s*\{.*?\}\n",
        "_MODELS = {\n"
        '    "stage1_1B_pt": "",\n'
        '    "stage2_1B_pt": "",\n'
        '    "stage1_6B_pt": "",\n'
        "}\n",
        text,
        flags=re.S,
    )

    fallback_defs = """

class FusedMLP(nn.Module):
    def __init__(self, in_features, hidden_features, activation=None, bias1=True, bias2=True, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DropoutAddRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, prenorm=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x, residual=None, prenorm=True):
        if residual is None:
            residual = x
        return self.norm(x), residual


OrderedDict = dict
"""

    if "class FusedMLP(nn.Module):" not in text:
        text = fallback_defs + "\n" + text

    return text


def build_inlined_internvideo_source(internvideo_code_path: str) -> str:
    """
    Build a self-contained InternVideo source blob for model.py.
    """
    code_root = Path(internvideo_code_path).expanduser().resolve()
    models_src = code_root / "models"
    if not models_src.is_dir():
        raise FileNotFoundError(
            f"InternVideo models directory not found: {models_src}. "
            "Expected --internvideo-code-path to point at InternVideo2/single_modality."
        )

    pos_embed_src = models_src / "pos_embed.py"
    teacher_src = models_src / "internvideo2_teacher.py"
    if not pos_embed_src.is_file():
        raise FileNotFoundError(f"Required InternVideo source not found: {pos_embed_src}")
    if not teacher_src.is_file():
        raise FileNotFoundError(f"Required InternVideo source not found: {teacher_src}")

    pos_text = pos_embed_src.read_text(errors="ignore")
    # Drop duplicate numpy import; model.py already imports numpy.
    pos_text = re.sub(r"^\s*import numpy as np\s*$", "", pos_text, flags=re.M)
    teacher_text = _sanitize_internvideo_teacher_source(teacher_src.read_text(errors="ignore"))

    return "\n\n".join([
        _sanitized_flash_attention_class_py(),
        pos_text,
        teacher_text,
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to training checkpoint (best.pt)")
    parser.add_argument("--output-dir", type=str, default="/home/shadeform/bittensor/internvideo2_trained_export")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature; defaults from best_temperature.txt if present else 1.0")
    parser.add_argument("--input-format", type=str, default="RGB", choices=["RGB", "BGR"])
    parser.add_argument("--internvideo-code-path", type=str, default="~/InternVideo/InternVideo2/single_modality")
    parser.add_argument("--name", type=str, default="BitmindModel")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.temperature is not None:
        temperature = float(args.temperature)
    else:
        temp_file = ckpt_path.parent / "best_temperature.txt"
        if temp_file.exists():
            temperature = float(temp_file.read_text().strip())
        else:
            temperature = 1.0

    print(f"[1/4] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_state_dict = ckpt.get("model_state_dict", ckpt)

    print("[2/4] Converting weights...")
    export_sd = convert_checkpoint_to_export_state(model_state_dict)

    print("[3/4] Writing model files...")
    inlined_source = build_inlined_internvideo_source(args.internvideo_code_path)
    save_file(export_sd, str(output_dir / "model.safetensors"))
    hidden_size = int(export_sd["classifier.weight"].shape[1])
    (output_dir / "model.py").write_text(
        build_model_py(
            temperature=temperature,
            input_format=args.input_format,
            inlined_internvideo_source=inlined_source,
        ).replace("__HIDDEN_SIZE__", str(hidden_size))
    )

    config_yaml = f"""name: "{args.name}"
version: "1.0.0"
modality: "video"

preprocessing:
  resize: [224, 224]
  max_frames: 16

model:
  num_classes: 2
  weights_file: "model.safetensors"
  temperature: {temperature}
  input_format: "{args.input_format}"
"""
    (output_dir / "model_config.yaml").write_text(config_yaml)

    print("[4/4] Creating zip...")
    zip_path = output_dir.parent / f"{output_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "model_config.yaml", "model_config.yaml")
        zf.write(output_dir / "model.py", "model.py")
        zf.write(output_dir / "model.safetensors", "model.safetensors")

    model_size_gb = (output_dir / "model.safetensors").stat().st_size / (1024**3)
    zip_size_gb = zip_path.stat().st_size / (1024**3)
    print("Export complete.")
    print(f"  dir: {output_dir}")
    print(f"  zip: {zip_path}")
    print(f"  model size: {model_size_gb:.2f} GB")
    print(f"  zip size:   {zip_size_gb:.2f} GB")


if __name__ == "__main__":
    main()
