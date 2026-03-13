import torch
import torch.nn as nn


class LogitCalibrator(nn.Module):
    """Learnable (T, b) calibration for logits (fake-bias only)."""

    def __init__(self, init_temperature: float = 1.0, init_fake_bias: float = 0.0):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))
        self.fake_bias = nn.Parameter(torch.tensor(init_fake_bias))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.exp(self.log_temperature).clamp(min=1e-3, max=100.0)
        bias = torch.stack([torch.zeros_like(self.fake_bias), self.fake_bias]).to(
            dtype=logits.dtype, device=logits.device
        )
        return logits / temperature + bias
