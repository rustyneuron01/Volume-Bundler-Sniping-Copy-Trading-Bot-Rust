import torch
import torch.nn as nn

from .calibration import LogitCalibrator


class SwinLargeVideo(nn.Module):
    """Swin-L (2D) with temporal average pooling for video."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        init_temperature: float = 1.0,
        init_bias: float = 0.0,
    ):
        super().__init__()

        try:
            import timm
        except Exception as exc:  # pragma: no cover - dependency missing
            raise RuntimeError("Swin-L requires timm. Install with: pip install timm") from exc

        self.backbone = timm.create_model(
            "swin_large_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
        )
        self.feature_dim = self.backbone.num_features

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W] RGB float in [0, 1]
        """
        x = (x - self.mean) / self.std
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.backbone(x)  # [B*T, F]
        feats = feats.view(b, t, -1).mean(dim=1)
        logits = self.classifier(feats)
        return self.calibrator(logits)


def create_swin_l(
    pretrained: bool = True,
    num_classes: int = 2,
    init_temperature: float = 1.0,
    init_bias: float = 0.0,
) -> SwinLargeVideo:
    return SwinLargeVideo(
        num_classes=num_classes,
        pretrained=pretrained,
        init_temperature=init_temperature,
        init_bias=init_bias,
    )
