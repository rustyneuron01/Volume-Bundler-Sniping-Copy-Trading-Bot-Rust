import torch
import torch.nn as nn

from .calibration import LogitCalibrator


class MViTv2Base(nn.Module):
    """MViTv2-Base classifier with in-model (T, b) calibration."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        init_temperature: float = 1.0,
        init_bias: float = 0.0,
    ):
        super().__init__()

        try:
            from torchvision.models.video import mvit_v2_b, MViT_V2_B_Weights
        except Exception as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "MViTv2 requires torchvision>=0.15. Install/upgrade torchvision."
            ) from exc

        weights = MViT_V2_B_Weights.KINETICS400_V1 if pretrained else None
        self.model = mvit_v2_b(weights=weights, num_classes=num_classes)

        if weights is not None:
            mean = weights.meta["mean"]
            std = weights.meta["std"]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        self.register_buffer("mean", torch.tensor(mean).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 1, 3, 1, 1), persistent=False)
        self.calibrator = LogitCalibrator(init_temperature, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W] RGB float in [0, 1]
        """
        x = (x - self.mean) / self.std
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] for torchvision
        logits = self.model(x)
        return self.calibrator(logits)


def create_mvit_v2_base(
    pretrained: bool = True,
    num_classes: int = 2,
    init_temperature: float = 1.0,
    init_bias: float = 0.0,
) -> MViTv2Base:
    return MViTv2Base(
        num_classes=num_classes,
        pretrained=pretrained,
        init_temperature=init_temperature,
        init_bias=init_bias,
    )
