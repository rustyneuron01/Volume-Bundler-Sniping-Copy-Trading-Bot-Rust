#!/usr/bin/env python3
"""
Export a trained MViTv2-L checkpoint (from train_mvitv2.py) to GasBench
custom PyTorch package format:
  - model_config.yaml
  - model.py
  - model.safetensors
  - <output_dir>.zip

The exported model.py depends on `timm` at runtime (standard library,
widely available in PyTorch environments).

Usage:
  python export_mvitv2.py --checkpoint checkpoints/mvitv2_large/best.pt
  python export_mvitv2.py --checkpoint checkpoints/mvitv2_large/best.pt --temperature 1.3
"""

import argparse
import zipfile
from pathlib import Path

import torch
from safetensors.torch import save_file


def build_model_py(temperature: float) -> str:
    """Generate the self-contained model.py for GAsBench evaluation."""
    return f'''#!/usr/bin/env python3
"""MViTv2-L + TemporalTransformer binary video detector for GAsBench.

Architecture:
  1. timm MViTv2-L backbone processes each frame independently → [B*T, 1152]
  2. TemporalTransformer aggregates across frames → [B, 512]
  3. fc_norm + classifier → [B, 2] logits

GAsBench sends: uint8 RGB [B, T, C, H, W] in range [0, 255]
GAsBench expects: float32 logits [B, 2] (it applies softmax internally)
"""
import torch
import torch.nn as nn
import timm


MVITV2_TIMM_NAME = "mvitv2_large_cls.fb_inw21k"
MVITV2_FEATURE_DIM = 1152
TEMPORAL_DIM = 512
TEMPORAL_HEADS = 8
TEMPORAL_LAYERS = 2


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_heads=8,
                 num_layers=2, max_frames=16, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        B, T, _ = x.shape
        x = self.proj(x)
        x = x + self.pos_embed[:, :T, :]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return self.norm(x[:, 0])


class MViTv2BinaryWrapper(nn.Module):
    """GAsBench inference wrapper with horizontal-flip TTA.

    Input:  [B, T, C, H, W] uint8 RGB (0-255), T=16
    Output: [B, 2] float32 logits (GAsBench applies softmax)

    TTA (test-time augmentation): processes both original and horizontally
    flipped video, averages logits. Costs 2x compute but boosts accuracy
    on symmetric artifacts and reduces prediction variance.
    """

    def __init__(self, hidden_size={TEMPORAL_DIM}, temperature={temperature}):
        super().__init__()

        self.backbone = timm.create_model(
            MVITV2_TIMM_NAME, pretrained=False, num_classes=0, global_pool="avg",
        )
        self.temporal = TemporalTransformer(
            input_dim=MVITV2_FEATURE_DIM, hidden_dim=hidden_size,
            num_heads=TEMPORAL_HEADS, num_layers=TEMPORAL_LAYERS,
        )
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)
        self.temperature = float(temperature)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def _forward_single(self, x):
        """Process a single [B, T, C, H, W] float normalized input."""
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        features = self.backbone(x)
        features = features.reshape(B, T, -1)
        pooled = self.temporal(features)
        return self.classifier(self.fc_norm(pooled))

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {{tuple(x.shape)}}")

        x = x.float() / 255.0
        x = (x - self.mean) / self.std

        logits_orig = self._forward_single(x)
        logits_flip = self._forward_single(x.flip(dims=[-1]))
        logits = (logits_orig + logits_flip) / 2.0
        return logits / max(self.temperature, 1e-6)


def load_model(weights_path, num_classes=2, temperature={temperature}):
    from safetensors.torch import load_file

    state_dict = load_file(weights_path)
    hidden_size = int(state_dict["classifier.weight"].shape[1])
    model = MViTv2BinaryWrapper(hidden_size=hidden_size, temperature=temperature)
    msg = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {{"mean", "std"}}
    missing = set(msg.missing_keys)
    unexpected = set(msg.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {{sorted(unexpected)}}")
    if not missing.issubset(allowed_missing):
        raise RuntimeError(f"Missing required keys: {{sorted(missing - allowed_missing)}}")
    print("load_state_dict:", msg)
    model.train(False)
    return model
'''


def convert_checkpoint_to_export_state(model_state_dict: dict) -> dict:
    """Convert train_mvitv2 checkpoint keys to export wrapper keys.

    Training checkpoint keys (shared temporal module):
      videomae.backbone.stages.0.blocks.0.norm1.weight   → backbone.stages.0.blocks.0.norm1.weight
      videomae.backbone.patch_embed.proj.weight           → backbone.patch_embed.proj.weight
      classifier_temporal.proj.weight                     → temporal.proj.weight
      classifier_temporal.cls_token                       → temporal.cls_token
      fc_norm.weight                                      → fc_norm.weight
      classifier.weight                                   → classifier.weight

    Note: In training, the temporal transformer is registered as 'classifier_temporal'
    (for head LR grouping) but in the export wrapper it's called 'temporal'.
    """
    out = {}
    backbone_count = 0
    temporal_count = 0

    for key, value in model_state_dict.items():
        if key.startswith("videomae.backbone."):
            out["backbone." + key[len("videomae.backbone."):]] = value
            backbone_count += 1
        elif key.startswith("classifier_temporal."):
            out["temporal." + key[len("classifier_temporal."):]] = value
            temporal_count += 1
        elif key.startswith("fc_norm.") or key.startswith("classifier."):
            out[key] = value

    if backbone_count == 0:
        raise ValueError(
            "No backbone weights found. Expected keys starting with 'videomae.backbone.'. "
            "Make sure --checkpoint points to train_mvitv2 best.pt."
        )
    if temporal_count == 0:
        raise ValueError(
            "No temporal transformer weights found. Expected keys starting with "
            "'classifier_temporal.'. Is this a train_mvitv2 checkpoint?"
        )

    required = ["classifier.weight", "classifier.bias", "fc_norm.weight", "fc_norm.bias"]
    missing = [k for k in required if k not in out]
    if missing:
        raise ValueError(f"Missing required classifier keys: {missing}")

    print(f"[export] Converted {backbone_count} backbone + {temporal_count} temporal + "
          f"{len(out) - backbone_count - temporal_count} head params")
    return out


def main():
    parser = argparse.ArgumentParser(description="Export MViTv2-L to GAsBench format")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to training checkpoint (best.pt)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: sibling of checkpoint)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature; reads best_temperature.txt if not set")
    parser.add_argument("--name", type=str, default="BitmindModel")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = ckpt_path.parent.parent / "mvitv2_export"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.temperature is not None:
        temperature = float(args.temperature)
    else:
        temp_file = ckpt_path.parent / "best_temperature.txt"
        if temp_file.exists():
            temperature = float(temp_file.read_text().strip())
            print(f"[temperature] Loaded from best_temperature.txt: {temperature}")
        else:
            temperature = 1.0
            print(f"[temperature] Using default: {temperature}")

    print(f"[1/4] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_state_dict = ckpt.get("model_state_dict", ckpt)

    print("[2/4] Converting weights...")
    export_sd = convert_checkpoint_to_export_state(model_state_dict)

    print("[3/4] Writing model files...")
    safetensors_path = output_dir / "model.safetensors"
    save_file(export_sd, str(safetensors_path))

    (output_dir / "model.py").write_text(build_model_py(temperature=temperature))

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
  input_format: "RGB"
"""
    (output_dir / "model_config.yaml").write_text(config_yaml)

    print("[4/4] Creating zip...")
    zip_path = output_dir.parent / f"{output_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(safetensors_path, "model.safetensors")
        zf.write(output_dir / "model.py", "model.py")
        zf.write(output_dir / "model_config.yaml", "model_config.yaml")

    model_size_gb = safetensors_path.stat().st_size / (1024 ** 3)
    zip_size_gb = zip_path.stat().st_size / (1024 ** 3)

    print(f"\nExport complete.")
    print(f"  dir:        {output_dir}")
    print(f"  zip:        {zip_path}")
    print(f"  model size: {model_size_gb:.2f} GB")
    print(f"  zip size:   {zip_size_gb:.2f} GB")
    print(f"  temperature: {temperature}")
    print(f"  4GB limit:  {'OK' if model_size_gb < 4.0 else 'OVER LIMIT!'}")

    if model_size_gb >= 4.0:
        print(f"\n  WARNING: Model exceeds 4GB limit!")
        print(f"  The model is {model_size_gb:.2f} GB in FP32.")
        print(f"  Consider FP16 conversion (expected ~{model_size_gb/2:.2f} GB)")


if __name__ == "__main__":
    main()
