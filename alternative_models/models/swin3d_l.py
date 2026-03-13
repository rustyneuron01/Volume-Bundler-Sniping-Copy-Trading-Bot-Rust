import torch
import torch.nn as nn

from .calibration import LogitCalibrator


class Swin3DLarge(nn.Module):
    """True 3D Swin-L backbone (mmaction2) with calibrated logits."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        drop_path_rate: float = 0.2,
        init_temperature: float = 1.0,
        init_bias: float = 0.0,
    ):
        super().__init__()

        try:
            from mmaction.models.backbones import SwinTransformer3D
        except Exception as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "Swin3D-L requires mmaction2. Install with: pip install mmaction2"
            ) from exc

        # Swin-L config (3D)
        self.backbone = SwinTransformer3D(
            patch_size=(2, 4, 4),
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=(8, 7, 7),
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            patch_norm=True,
        )
        self.feature_dim = 1536

        # ImageNet normalization
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, num_classes),
        )
        self.calibrator = LogitCalibrator(init_temperature, init_bias)

        if pretrained:
            # mmaction2 pretrained weights must be loaded manually by the user.
            # Keep this hook for external checkpoint loading.
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W] RGB float in [0, 1]
        """
        x = (x - self.mean) / self.std
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        feats = self.backbone(x)

        if feats.dim() == 5:
            feats = feats.mean(dim=[2, 3, 4])
        elif feats.dim() == 3:
            feats = feats.mean(dim=1)
        elif feats.dim() == 2:
            pass
        else:
            raise ValueError(f"Unexpected backbone output shape: {feats.shape}")

        logits = self.classifier(feats)
        return self.calibrator(logits)


def create_swin3d_l(
    pretrained: bool = True,
    num_classes: int = 2,
    drop_path_rate: float = 0.2,
    init_temperature: float = 1.0,
    init_bias: float = 0.0,
) -> Swin3DLarge:
    return Swin3DLarge(
        num_classes=num_classes,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
        init_temperature=init_temperature,
        init_bias=init_bias,
    )
