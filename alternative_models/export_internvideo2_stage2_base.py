#!/usr/bin/env python3
"""
Export InternVideo2 Stage2 1B base checkpoint into gasbench safetensors package.

This exporter targets local benchmarking and depends on a local InternVideo clone:
  ~/InternVideo/InternVideo2/single_modality

Output package:
  - model_config.yaml
  - model.py
  - model.safetensors
  - zip archive with those 3 files at root
"""

import argparse
from pathlib import Path
import zipfile

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file


def build_model_py(temperature: float, input_format: str) -> str:
    return f'''#!/usr/bin/env python3
import importlib
import os
import sys
import types

import torch
import torch.nn as nn


def _install_flash_attn_fallback_stubs():
    mlp_mod = types.ModuleType("flash_attn.modules.mlp")

    class FusedMLP(nn.Module):
        def __init__(self, in_features, hidden_features, activation=None, bias1=True, bias2=True):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_features, in_features, bias=bias2)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    mlp_mod.FusedMLP = FusedMLP
    sys.modules["flash_attn.modules.mlp"] = mlp_mod

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


class InternVideo2BinaryWrapper(nn.Module):
    def __init__(self, temperature={temperature}, input_format="{input_format}"):
        super().__init__()
        code_root = os.path.expanduser("~/InternVideo/InternVideo2/single_modality")
        models_dir = os.path.join(code_root, "models")
        sys.path.insert(0, code_root)
        _install_flash_attn_fallback_stubs()

        models_pkg = types.ModuleType("models")
        models_pkg.__path__ = [models_dir]
        sys.modules["models"] = models_pkg
        teacher_mod = importlib.import_module("models.internvideo2_teacher")
        InternVideo2 = teacher_mod.InternVideo2

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
        self.adapter_head = nn.Linear(768, 2)
        self.temperature = float(temperature)
        self.input_format = input_format.upper()

    def forward(self, x):
        # gasbench video input: [B, T, C, H, W] uint8
        if x.dim() != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {{tuple(x.shape)}}")
        x = x.float() / 255.0
        if self.input_format == "BGR":
            x = x[:, :, [2, 1, 0], :, :]
        x = x.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W]
        if x.shape[2] > 8:
            x = x[:, :, :8]
        elif x.shape[2] < 8:
            pad = 8 - x.shape[2]
            x = torch.cat([x, x[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2)
        _, feature_768 = self.backbone(x)
        logits = self.adapter_head(feature_768)
        return logits / max(self.temperature, 1e-6)


def load_model(weights_path: str, num_classes: int = 2, temperature: float = {temperature}, input_format: str = "{input_format}"):
    from safetensors.torch import load_file

    model = InternVideo2BinaryWrapper(temperature=temperature, input_format=input_format)
    state_dict = load_file(weights_path)
    msg = model.load_state_dict(state_dict, strict=False)
    print("load_state_dict:", msg)
    model.eval()
    return model
'''


def _interpolate_stage2_pos_embed_to_t8(module_sd: dict) -> None:
    """Interpolate vision_encoder.pos_embed from T=4 to T=8 when needed."""
    key = "vision_encoder.pos_embed"
    if key not in module_sd:
        return
    src = module_sd[key]  # [1, 1+T*H*W, C]
    if src.ndim != 3 or src.shape[1] == 2049:
        return
    if src.shape[1] != 1025:
        return

    cls_src = src[:, :1, :]
    tok_src = src[:, 1:, :]
    c = tok_src.shape[-1]
    t_src = 4
    s_src = int((tok_src.shape[1] // t_src) ** 0.5)
    t_dst = 8
    s_dst = s_src
    tok_src = tok_src.reshape(1, t_src, s_src, s_src, c).permute(0, 4, 1, 2, 3)
    tok_src = torch.nn.functional.interpolate(
        tok_src, size=(t_dst, s_dst, s_dst), mode="trilinear", align_corners=False
    )
    tok_src = tok_src.permute(0, 2, 3, 4, 1).reshape(1, t_dst * s_dst * s_dst, c)
    module_sd[key] = torch.cat([cls_src, tok_src], dim=1)


def convert_checkpoint_to_export_state(module_sd: dict, seed: int) -> dict:
    _interpolate_stage2_pos_embed_to_t8(module_sd)
    out = {}
    # Keep only vision_encoder weights and strip prefix.
    for key, value in module_sd.items():
        if not key.startswith("vision_encoder."):
            continue
        if "clip_decoder" in key or "final_clip_decoder" in key:
            continue
        if "clip_pos_embed" in key or "clip_img_pos_embed" in key or "img_pos_embed" in key:
            continue
        out["backbone." + key.replace("vision_encoder.", "", 1)] = value

    # Add minimal binary adapter head for gasbench scoring.
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out["adapter_head.weight"] = torch.empty((2, 768), dtype=torch.float32)
    torch.nn.init.normal_(out["adapter_head.weight"], mean=0.0, std=0.02, generator=gen)
    out["adapter_head.bias"] = torch.zeros(2, dtype=torch.float32)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="OpenGVLab/InternVideo2-Stage2_1B-224p-f4")
    parser.add_argument("--filename", type=str, default="InternVideo2-stage2_1b-224p-f4.pt")
    parser.add_argument("--output-dir", type=str, default="/home/shadeform/bittensor/internvideo2_stage2_base_export")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--input-format", type=str, default="RGB", choices=["RGB", "BGR"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Downloading checkpoint from Hugging Face...")
    ckpt_path = hf_hub_download(repo_id=args.repo_id, filename=args.filename, token=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    module_sd = ckpt["module"]

    print("[2/5] Converting checkpoint to export state...")
    export_sd = convert_checkpoint_to_export_state(module_sd, args.seed)

    print("[3/5] Writing model.safetensors...")
    save_file(export_sd, str(output_dir / "model.safetensors"))

    print("[4/5] Writing model.py and model_config.yaml...")
    (output_dir / "model.py").write_text(build_model_py(args.temperature, args.input_format))
    config_yaml = f"""name: "internvideo2-stage2-1b-base"
version: "1.0.0"
modality: "video"

preprocessing:
  resize: [224, 224]
  max_frames: 8

model:
  num_classes: 2
  weights_file: "model.safetensors"
  temperature: {args.temperature}
  input_format: "{args.input_format}"
"""
    (output_dir / "model_config.yaml").write_text(config_yaml)

    print("[5/5] Creating zip (root files only)...")
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

