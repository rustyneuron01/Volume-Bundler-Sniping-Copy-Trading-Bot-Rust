import torch
import torch.nn as nn

from .calibration import LogitCalibrator


class VideoMAEv2ViTL(nn.Module):
    """
    VideoMAE v2 ViT-L classifier with in-model (T, b) calibration.
    Uses Hugging Face transformers for pretrained weights.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        model_id: str = "MCG-NJU/videomae-v2-large",
        init_temperature: float = 1.0,
        init_bias: float = 0.0,
    ):
        super().__init__()

        try:
            from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
        except Exception as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "VideoMAE v2 requires transformers. Install with: pip install transformers"
            ) from exc

        self.processor = VideoMAEImageProcessor.from_pretrained(model_id)
        self.register_buffer(
            "mean",
            torch.tensor(self.processor.image_mean).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(self.processor.image_std).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        if pretrained:
            self.model = VideoMAEForVideoClassification.from_pretrained(
                model_id,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            self.model = VideoMAEForVideoClassification.from_pretrained(
                model_id,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
            self.model.init_weights()

        self.calibrator = LogitCalibrator(init_temperature, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W] RGB float in [0, 1]
        """
        # Normalize using VideoMAE defaults
        x = (x - self.mean) / self.std
        outputs = self.model(pixel_values=x, return_dict=True)
        logits = outputs.logits
        return self.calibrator(logits)


def create_videomae_v2_vitl(
    pretrained: bool = True,
    num_classes: int = 2,
    model_id: str = "MCG-NJU/videomae-v2-large",
    init_temperature: float = 1.0,
    init_bias: float = 0.0,
) -> VideoMAEv2ViTL:
    return VideoMAEv2ViTL(
        num_classes=num_classes,
        pretrained=pretrained,
        model_id=model_id,
        init_temperature=init_temperature,
        init_bias=init_bias,
    )
