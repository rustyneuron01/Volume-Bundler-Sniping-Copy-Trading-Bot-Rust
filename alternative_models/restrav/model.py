"""
ReStraV-style video detector: frozen DINOv2 + trajectory geometry (21-D) + MLP.

Architecture:
  Input [B, T, C, H, W] float32 (0-255, from gasbench)
  → resize to 224×224, /255, ImageNet normalize
  → Frozen DINOv2 ViT-S/14 per-frame → [B, T, D] (D=384)
  → Trajectory geometry: stepwise distance d_i, curvature θ_i → 21-D per video
  → (feat - mean) / (std + 1e-8)  [buffers set by calibration pass]
  → MLP(21 → 64 → 32 → 2) → logits [B, 2]

Only the MLP is trained. Backbone is always frozen.
gasbench: load_model(weights_path, **kwargs), forward(x) → [B, 2] logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ReStraV 21-D: first 7 stepwise distances, first 6 curvatures, 8 stats (mean/min/max/var for d and θ)
NUM_GEOM_FEATURES = 21


def compute_trajectory_geometry(Z):
    """Z: [B, T, D]. Returns d [B, T-1], theta [B, T-2] in degrees."""
    delta = Z[:, 1:, :] - Z[:, :-1, :]
    d = delta.norm(dim=-1)
    # curvature: angle between consecutive displacement vectors
    cos_sim = F.cosine_similarity(delta[:, :-1, :], delta[:, 1:, :], dim=-1)
    theta_rad = torch.acos(cos_sim.clamp(-1.0, 1.0))
    theta_deg = torch.rad2deg(theta_rad)
    return d, theta_deg


def trajectory_to_21d(d, theta):
    """d [B, T-1], theta [B, T-2]. Returns [B, 21]."""
    # first 7 distances, first 6 curvatures
    d7 = d[:, :7]
    t6 = theta[:, :6]
    # 8 stats: mean, min, max, var for d and for theta
    mu_d = d.mean(dim=-1)
    mn_d = d.amin(dim=-1)
    mx_d = d.amax(dim=-1)
    var_d = d.var(dim=-1, unbiased=False)
    mu_t = theta.mean(dim=-1)
    mn_t = theta.amin(dim=-1)
    mx_t = theta.amax(dim=-1)
    var_t = theta.var(dim=-1, unbiased=False)
    stats = torch.stack([mu_d, mn_d, mx_d, var_d, mu_t, mn_t, mx_t, var_t], dim=1)
    return torch.cat([d7, t6, stats], dim=1)


class ReStraVDetector(nn.Module):
    """Frozen DINOv2 + 21-D trajectory geometry + MLP. Only MLP is trainable."""

    BACKBONE_MAP = {
        "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
        "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
        "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    }

    ALLOWED_KWARGS = {"backbone", "mlp_h1", "mlp_h2"}

    def __init__(
        self,
        num_classes=2,
        backbone="dinov2_vits14",
        pretrained=True,
        mlp_h1=64,
        mlp_h2=32,
    ):
        super().__init__()
        timm_name = self.BACKBONE_MAP.get(backbone, backbone)
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="token",
            img_size=224,
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat_dim = self.backbone(dummy).shape[-1]
        self.feat_dim = feat_dim

        self.register_buffer(
            "mean",
            torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        # 21-D feature normalization (set by calibration pass in training)
        self.register_buffer("feat_mean", torch.zeros(NUM_GEOM_FEATURES))
        self.register_buffer("feat_std", torch.ones(NUM_GEOM_FEATURES))

        self.classifier = nn.Sequential(
            nn.Linear(NUM_GEOM_FEATURES, mlp_h1),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_h1, mlp_h2),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_h2, num_classes),
        )

    def get_geometry_features(self, x):
        """Return [B, 21] trajectory geometry features (before normalization). Used for calibration."""
        B, T, C, H, W = x.shape
        x = x.float() / 255.0
        if H != 224 or W != 224:
            x = F.interpolate(
                x.view(B * T, C, H, W),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).view(B, T, C, 224, 224)
        x = (x - self.mean) / self.std
        x = x.view(B * T, C, 224, 224)
        with torch.no_grad():
            z = self.backbone(x)
        z = z.view(B, T, -1)
        d, theta = compute_trajectory_geometry(z)
        return trajectory_to_21d(d, theta)

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] float32, 0-255 (e.g. from gasbench or train loader)
        Returns:
            logits: [B, num_classes]
        """
        feat_21 = self.get_geometry_features(x)
        feat_norm = (feat_21 - self.feat_mean) / (self.feat_std + 1e-8)
        logits = self.classifier(feat_norm)
        return logits

    def set_feature_normalization(self, mean, std):
        """Set 21-D normalization (call after calibration pass)."""
        with torch.no_grad():
            self.feat_mean.copy_(torch.as_tensor(mean, dtype=self.feat_mean.dtype, device=self.feat_mean.device))
            self.feat_std.copy_(torch.as_tensor(std, dtype=self.feat_std.dtype, device=self.feat_std.device) + 1e-8)


def create_model(num_classes=2, backbone="dinov2_vits14", pretrained=True, **kwargs):
    return ReStraVDetector(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        **{k: v for k, v in kwargs.items() if k in ReStraVDetector.ALLOWED_KWARGS},
    )


def load_model(weights_path, num_classes=2, backbone="dinov2_vits14", **kwargs):
    """Load for gasbench: model = module.load_model(weights_path, **model_config)."""
    kwargs.pop("dtype", None)
    kwargs.pop("bf16", None)
    kwargs.pop("config", None)
    model = ReStraVDetector(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=False,
        **{k: v for k, v in kwargs.items() if k in ReStraVDetector.ALLOWED_KWARGS},
    )
    if str(weights_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


if __name__ == "__main__":
    print("Testing ReStraVDetector...")
    model = create_model()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total / 1e6:.2f}M  Trainable: {trainable / 1e3:.1f}K")
    x = torch.randint(0, 256, (2, 24, 3, 224, 224), dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
    print(f"Input: {x.shape}  Output: {logits.shape}")
    x518 = torch.randint(0, 256, (2, 24, 3, 518, 518), dtype=torch.float32)
    with torch.no_grad():
        logits518 = model(x518)
    print(f"Input 518: {x518.shape}  Output: {logits518.shape}")
