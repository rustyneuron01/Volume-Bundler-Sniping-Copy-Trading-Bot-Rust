#!/usr/bin/env python3
"""
Export VideoMAE-v2 for gasbench/Subnet 34 submission.
Format: https://github.com/bitmind-ai/gasbench/blob/main/docs/Safetensors.md
100% self-contained - validators only need: torch, safetensors
"""

import argparse
from pathlib import Path
import zipfile
import torch
from safetensors.torch import save_file


def create_standalone_model(temperature, hidden_size=1408, input_format="BGR"):
    """
    Create the complete standalone model code with embedded VideoMAE architecture.
    This code has ZERO external dependencies except torch.
    """
    
    code = '''#!/usr/bin/env python3
"""
Standalone VideoMAE-v2 Giant Classifier
100% self-contained - validators need only PyTorch (no other dependencies)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class DropPath(nn.Module):
    """Stochastic Depth (Drop Path)"""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias, requires_grad=False),
                 self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., 
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                             attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Video to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, 
                 num_frames=16, tubelet_size=2):
        super().__init__()
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VideoMAEv2Backbone(nn.Module):
    """VideoMAE-v2 Giant backbone"""
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=''' + str(hidden_size) + ''',
                 depth=40, num_heads=16, mlp_ratio=4.3637, num_frames=16, tubelet_size=2):
        super().__init__()
        
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim, num_frames, tubelet_size
        )
        
        num_patches = (img_size // patch_size) ** 2 * (num_frames // tubelet_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)
        
        dpr = [x.item() for x in torch.linspace(0, 0.25, depth, device='cpu')]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, 
                  drop_path=dpr[i], norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        
        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        # Match HF: norm applied to first token only, returns [B, embed_dim]
        return self.norm(x[:, 0])


class VideoDetector(nn.Module):
    """Complete VideoMAE-v2 detector - 100% standalone"""
    def __init__(self, temperature=''' + str(temperature) + ''', input_format="''' + input_format + '''"):
        super().__init__()
        self.temperature = temperature
        self.input_format = input_format.upper()
        
        # VideoMAE-v2 Giant
        self.backbone = VideoMAEv2Backbone(
            img_size=224, patch_size=14, embed_dim=''' + str(hidden_size) + ''',
            depth=40, num_heads=16, num_frames=16, tubelet_size=2
        )
        
        # Classification head
        self.fc_norm = nn.LayerNorm(''' + str(hidden_size) + ''')
        self.classifier = nn.Linear(''' + str(hidden_size) + ''', 2)
        
        # Normalization (ImageNet RGB format)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
    
    def forward(self, x):
        """
        Args:
            x: [B, T, 3, H, W] uint8 video (0-255)
               Format determined by input_format (set during export)
               BGR = cv2.VideoCapture format (gasbench default)
               RGB = standard PyTorch/PIL format
        Returns:
            logits: [B, 2]
        """
        # Validate input shape - gasbench should send [B, T, C, H, W]
        if x.dim() == 4:
            # Emergency fallback: assume [B, C, H, W] is actually [B*T, C, H, W]
            # Try to recover temporal dimension
            if x.shape[0] >= 16:
                # Reshape [B*T, C, H, W] -> [B, T, C, H, W]
                batch_with_time = x.shape[0]
                if batch_with_time % 16 == 0:
                    B = batch_with_time // 16
                    x = x.view(B, 16, x.shape[1], x.shape[2], x.shape[3])
                else:
                    # Can't recover, use single frame
                    x = x.unsqueeze(1)
            else:
                # Single frame [B, 3, H, W] - add temporal dimension
                x = x.unsqueeze(1)  # [B, 3, H, W] -> [B, 1, 3, H, W]
        
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input [B, T, C, H, W], got {x.dim()}D with shape {x.shape}")
        
        # Normalize
        x = x.float() / 255.0
        
        # Convert to RGB if needed
        if self.input_format == "BGR":
            x = x[:, :, [2, 1, 0], :, :]  # BGR -> RGB
        
        x = (x - self.mean) / self.std
        
        # Transpose to [B, C, T, H, W] for VideoMAE
        x = x.permute(0, 2, 1, 3, 4)
        
        # Pad/trim to 16 frames
        if x.shape[2] != 16:
            if x.shape[2] > 16:
                x = x[:, :, :16]
            else:
                pad = 16 - x.shape[2]
                last = x[:, :, -1:].repeat(1, 1, pad, 1, 1)
                x = torch.cat([x, last], dim=2)
        
        # Backbone - returns norm(x[:, 0]) already [B, embed_dim]
        pooled = self.backbone(x)
        
        # Classifier
        pooled = self.fc_norm(pooled)
        logits = self.classifier(pooled)
        logits = logits / max(self.temperature, 1e-6)
        
        return logits


def load_model(weights_path: str, num_classes: int = 2, temperature: float = ''' + str(temperature) + ''', input_format: str = "''' + input_format + '''"):
    """
    Load model with weights - REQUIRED by gasbench specification.
    
    Args:
        weights_path: Path to model.safetensors
        num_classes: Number of classes (from config, always 2)
        temperature: Temperature scaling factor
        input_format: Color format (set during export, matches model_config.yaml)
        
    Returns:
        Loaded PyTorch model ready for inference
    """
    from safetensors.torch import load_file
    
    model = VideoDetector(temperature=temperature, input_format=input_format)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


if __name__ == "__main__":
    print("VideoMAE-v2 Giant Standalone Detector")
    print("Dependencies: torch only")
    model = load_model()
    print(f"✓ Model loaded (temperature={model.temperature})")
    
    # Test
    x = torch.randint(0, 255, (1, 16, 3, 224, 224), dtype=torch.uint8)
    with torch.no_grad():
        out = model(x)
    print(f"✓ Test passed: {out.shape}")
'''
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="videomae_selfcontained")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--input-format", type=str, default="RGB", choices=["BGR", "RGB"], 
                       help="Input color format: BGR (gasbench default) or RGB")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    
    state_dict = {k: v for k, v in state_dict.items() if torch.is_tensor(v)}

    # Remap backbone weights from training wrapper to standalone backbone
    # Training wrapper stores backbone under "videomae.*" while standalone uses "backbone.*"
    remapped = {}
    remap_count = 0
    for k, v in state_dict.items():
        if k.startswith("videomae.model."):
            remapped["backbone." + k[len("videomae.model."):]] = v
            remap_count += 1
        elif k.startswith("videomae."):
            remapped["backbone." + k[len("videomae."):]] = v
            remap_count += 1
        else:
            remapped[k] = v
    if remap_count > 0:
        state_dict = remapped
        print(f"[1/4] Remapped {remap_count} backbone keys (videomae.* → backbone.*)")
    else:
        if not any(k.startswith("backbone.") for k in state_dict.keys()):
            print("[1/4] WARNING: No backbone weights found; export may be randomly initialized.")
    
    # Load pos_embed from HF pretrained model if not in checkpoint
    if "backbone.pos_embed" not in state_dict:
        pos_embed_path = Path(args.checkpoint).parent / "pos_embed.pt"
        if pos_embed_path.exists():
            pos_embed = torch.load(pos_embed_path, map_location="cpu", weights_only=True)
            state_dict["backbone.pos_embed"] = pos_embed
            print(f"[1/4] Loaded pos_embed from {pos_embed_path} (shape: {pos_embed.shape})")
        else:
            print(f"[1/4] WARNING: pos_embed not found in checkpoint or {pos_embed_path}")
            print(f"[1/4]   Run: python -c \"import torch,os; os.environ['ACCELERATE_USE_CPU']='1'; from train_videomae_v2 import VideoMAEv2ForClassification; m=VideoMAEv2ForClassification('OpenGVLab/VideoMAEv2-giant',2); torch.save(m.videomae.model.pos_embed.data.cpu(), '{pos_embed_path}')\"")
            print(f"[1/4]   Then re-run export.")
    
    if args.bf16:
        state_dict = {k: v.to(dtype=torch.bfloat16) for k, v in state_dict.items()}

    # Get hidden size from state dict (try multiple key patterns)
    hidden_size = None
    detection_keys = [
        'fc_norm.weight',           # Training wrapper (current)
        'videomae.fc_norm.weight',  # Alternative wrapper
        'videomae.model.norm.weight',  # Standalone backbone
        'backbone.norm.weight',     # Direct backbone
        'norm.weight',              # Minimal structure
    ]
    
    for key in detection_keys:
        if key in state_dict:
            hidden_size = state_dict[key].shape[0]
            print(f"[2/4] Detected hidden_size: {hidden_size} (from {key})")
            break
    
    if hidden_size is None:
        hidden_size = 1408  # VideoMAE-v2 Giant default fallback
        print(f"[2/4] Using default hidden_size: {hidden_size} (no norm key found)")
    
    # Validate detected hidden_size against classifier
    if 'classifier.weight' in state_dict:
        classifier_in = state_dict['classifier.weight'].shape[1]
        if classifier_in != hidden_size:
            print(f"  ⚠️  WARNING: Classifier input ({classifier_in}) != hidden_size ({hidden_size})")
            print(f"  Using classifier input size: {classifier_in}")
            hidden_size = classifier_in

    # Get temperature
    temperature = args.temperature
    if temperature is None:
        temp_path = Path(args.checkpoint).parent / "best_temperature.txt"
        if temp_path.exists():
            temperature = float(temp_path.read_text().strip())
        else:
            temperature = 1.6

    print(f"[3/5] Creating model_config.yaml...")
    config_yaml = f"""name: "videomae-v2-giant"
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

    print(f"[4/5] Creating standalone model.py...")
    model_code = create_standalone_model(temperature, hidden_size, args.input_format)
    (output_dir / "model.py").write_text(model_code)

    print(f"[5/5] Saving weights as model.safetensors...")
    save_file(state_dict, str(output_dir / "model.safetensors"))
    
    size_gb = (output_dir / "model.safetensors").stat().st_size / (1024 ** 3)
    print(f"  Model size: {size_gb:.2f} GB")
    
    if size_gb > 4.0:
        print(f"  ⚠️  WARNING: Model exceeds 4GB limit!")

    # Create zip per gasbench spec
    zip_path = output_dir.parent / f"{output_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "model_config.yaml", "model_config.yaml")
        zf.write(output_dir / "model.py", "model.py")
        zf.write(output_dir / "model.safetensors", "model.safetensors")

    zip_size_gb = zip_path.stat().st_size / (1024 ** 3)

    print(f"\n✅ Export complete (gasbench/SN34 format)!")
    print(f"  Directory: {output_dir}/")
    print(f"  Zip: {zip_path} ({zip_size_gb:.2f} GB)")
    print(f"  Temperature: {temperature}")
    print(f"  Input format: {args.input_format}")
    print(f"\n📦 Validators install:")
    print(f"  pip install torch safetensors")
    print(f"\n✅ 100% self-contained:")
    print(f"  - All VideoMAE architecture embedded")
    print(f"  - No transformers needed")
    print(f"  - No HuggingFace downloads")
    print(f"  - No trust_remote_code")
    print(f"\n🚀 Submit to Subnet 34:")
    print(f"  gascli d push --video-model {zip_path}")


if __name__ == "__main__":
    main()
