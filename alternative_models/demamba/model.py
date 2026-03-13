"""
CLIP ViT-L/14 + DeMamba Video Fake/Real Detector

Based on DeMamba (NeurIPS 2024):
  "AI-Generated Video Detection on Million-Scale GenVideo Benchmark"
  https://github.com/chenhaoxing/DeMamba

Architecture:
  Input [B, T, C, H, W] float32 (0-255, uint8 range from gasbench)
  -> /255 + CLIP normalization
  -> Per-frame CLIP ViT-L/14 backbone -> [B*T, 257, 1024] (CLS + 256 patches)
  -> Project to 512d
  -> CLS tokens: mean pool across frames -> global_feat [B, 512]
  -> Patch tokens: spatial grouping (fusing_ratios=2, 64 groups of 4 patches)
    -> Zigzag reorder within each group (preserves spatial adjacency)
    -> Flatten temporal + spatial: [B*64, 32, 512]
    -> Bidirectional Mamba SSM: captures spatio-temporal inconsistencies
    -> Mean pool per group -> [B, 64*512]
    -> LayerNorm
  -> Concat [global_feat, mamba_feat] -> [B, 33280]
  -> Linear(33280, 2) -> [B, 2] logits

Why DeMamba:
  - Purpose-built for AI-generated video detection (not repurposed from classification)
  - #1 on GenVidBench (AAAI 2026): 85.47% Top-1, 99.28% AUROC cross-generator
  - Mamba SSM has O(n) complexity -> fast inference
  - Bidirectional scan captures both forward/backward temporal patterns
  - Zigzag spatial grouping captures local spatial consistency within frames

gasbench compatibility:
  - load_model(weights_path, **kwargs) required by custom_model_loader.py
  - forward() accepts [B, T, C, H, W] float32 (0-255)
  - Returns logits [B, 2] (gasbench applies softmax in process_model_output)
  - No getattr/setattr calls (passes static analysis)

CLIP ViT-L/14 @ 224x224:
  - 16x16 = 256 patches + 1 CLS = 257 tokens
  - Feature dimension: 1024
  - Contrastive pretraining on 400M image-text pairs
"""

import math
from dataclasses import dataclass, field
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# ============================================================================
# Parallel Scan (PScan) for Mamba SSM
# Adapted from: https://github.com/alxndrTL/mamba.py (MIT License)
# ============================================================================

def _npo2(length):
    return 2 ** math.ceil(math.log2(length))


def _pad_npo2(X):
    pad_len = _npo2(X.size(1)) - X.size(1)
    if pad_len == 0:
        return X
    return F.pad(X, (0, 0, 0, 0, 0, pad_len))


class PScan(torch.autograd.Function):
    @staticmethod
    def _pscan_fwd(A, X):
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))
        Aa, Xa = A, X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])
            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]
        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])
            Xa[:, :, 3].add_(
                Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1]))
            )
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return
        Aa = A[:, :, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
        Xa = X[:, :, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])
        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 2 ** k - 1:L:2 ** k]
            Xa = X[:, :, 2 ** k - 1:L:2 ** k]
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)
            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def _pscan_rev(A, X):
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))
        Aa, Xa = A, X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)
            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])
            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]
        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])
            Xa[:, :, 0].add_(
                Aa[:, :, 0].mul(
                    Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))
                )
            )
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return
        Aa = A[:, :, 0:L:2 ** (num_steps - 2)]
        Xa = X[:, :, 0:L:2 ** (num_steps - 2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])
        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 0:L:2 ** k]
            Xa = X[:, :, 0:L:2 ** k]
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)
            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        L = X_in.size(1)
        if L == _npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            A = _pad_npo2(A_in)
            X = _pad_npo2(X_in)
        A = A.transpose(2, 1)
        X = X.transpose(2, 1)
        PScan._pscan_fwd(A, X)
        ctx.save_for_backward(A_in, X)
        return X.transpose(2, 1)[:, :L]

    @staticmethod
    def backward(ctx, grad_output_in):
        A_in, X = ctx.saved_tensors
        L = grad_output_in.size(1)
        if L == _npo2(L):
            grad_output = grad_output_in.clone()
        else:
            grad_output = _pad_npo2(grad_output_in)
            A_in = _pad_npo2(A_in)
        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1)
        A = F.pad(A_in[:, :, 1:], (0, 0, 0, 1))
        PScan._pscan_rev(A, grad_output)
        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])
        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]


pscan = PScan.apply


# ============================================================================
# Mamba SSM Components (from DeMamba, NeurIPS 2024)
# ============================================================================

class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        if keep > 0.0:
            mask.div_(keep)
        return x * mask


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        ) * self.weight


@dataclass
class MambaConfig:
    d_model: int = 512
    dt_rank: Union[int, str] = "auto"
    d_state: int = 16
    expand_factor: int = 2
    d_conv: int = 4
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    drop_prob: float = 0.1
    bias: bool = False
    conv_bias: bool = True
    bimamba: bool = True
    use_pscan: bool = True
    d_inner: int = field(init=False)

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


class MambaBlock(nn.Module):
    """Bidirectional Mamba block for selective state space modeling."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        self.in_proj = nn.Linear(
            config.d_model, 2 * config.d_inner, bias=config.bias
        )
        self.conv1d = nn.Conv1d(
            config.d_inner, config.d_inner,
            kernel_size=config.d_conv, bias=config.conv_bias,
            groups=config.d_inner, padding=config.d_conv - 1,
        )
        self.x_proj = nn.Linear(
            config.d_inner, config.dt_rank + 2 * config.d_state, bias=False
        )
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(config.d_inner)
            * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(
            1, config.d_state + 1, dtype=torch.float32
        ).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.out_proj = nn.Linear(
            config.d_inner, config.d_model, bias=config.bias
        )

        self.bimamba = config.bimamba
        if self.bimamba:
            A_b = torch.arange(
                1, config.d_state + 1, dtype=torch.float32
            ).repeat(config.d_inner, 1)
            self.A_b_log = nn.Parameter(torch.log(A_b))
            self.conv1d_b = nn.Conv1d(
                config.d_inner, config.d_inner,
                kernel_size=config.d_conv, bias=config.conv_bias,
                groups=config.d_inner, padding=config.d_conv - 1,
            )
            self.x_proj_b = nn.Linear(
                config.d_inner, config.dt_rank + 2 * config.d_state, bias=False
            )
            self.dt_proj_b = nn.Linear(
                config.dt_rank, config.d_inner, bias=True
            )
            self.D_b = nn.Parameter(torch.ones(config.d_inner))

    def _scan(self, x, delta, A, B, C, D):
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        BX = deltaB * x.unsqueeze(-1)
        if self.config.use_pscan:
            hs = pscan(deltaA, BX)
        else:
            _, L, _ = x.shape
            h = torch.zeros(
                x.size(0), self.config.d_inner, self.config.d_state,
                device=x.device, dtype=x.dtype,
            )
            hs_list = []
            for t in range(L):
                h = deltaA[:, t] * h + BX[:, t]
                hs_list.append(h)
            hs = torch.stack(hs_list, dim=1)
        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        return y + D * x

    def _ssm(self, x):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        deltaBC = self.x_proj(x)
        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(delta))
        y = self._scan(x, delta, A, B, C, D)

        if self.bimamba:
            x_b = x.flip([1])
            A_b = -torch.exp(self.A_b_log.float())
            D_b = self.D_b.float()
            deltaBC_b = self.x_proj_b(x_b)
            delta_b, B_b, C_b = torch.split(
                deltaBC_b,
                [self.config.dt_rank, self.config.d_state,
                 self.config.d_state],
                dim=-1,
            )
            delta_b = F.softplus(self.dt_proj_b(delta_b))
            y_b = self._scan(x_b, delta_b, A_b, B_b, C_b, D_b)
            y = y + y_b.flip([1])

        return y

    def forward(self, x):
        _, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = F.silu(x)
        y = self._ssm(x)
        z = F.silu(z)
        return self.out_proj(y * z)


class MambaResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)
        self.drop_path = DropPath(drop_prob=config.drop_prob)

    def forward(self, x):
        return self.drop_path(self.mixer(self.norm(x))) + x


# ============================================================================
# DeMamba Spatial-Temporal Reordering
# Zigzag pattern preserves spatial adjacency for Mamba's sequential processing
# ============================================================================

def _create_zigzag_index(N, device):
    """Create zigzag traversal index for NxN grid."""
    new_order = []
    for col in range(N):
        if col % 2 == 0:
            new_order.extend(range(col, N * N, N))
        else:
            new_order.extend(range(col + N * (N - 1), col - 1, -N))
    return torch.tensor(new_order, device=device)


def _reorder_patches(data, N):
    """Reorder patches in zigzag pattern within each spatial group.

    Args:
        data: [B, T, N*N, C] tensor of patch features
        N: fusing_ratios (side length of spatial group)
    Returns:
        Reordered tensor with same shape
    """
    device = data.device
    idx = _create_zigzag_index(N, device)
    B, T, P, C = data.shape
    index = idx.repeat(B, T, 1).unsqueeze(-1)
    return torch.gather(data, 2, index.expand_as(data))


# ============================================================================
# CLIP ViT-L/14 + DeMamba Detector
# ============================================================================

class CLIPDeMambaDetector(nn.Module):
    """CLIP ViT-L/14 with DeMamba detection head for video fake/real detection.

    The DeMamba module uses bidirectional Mamba SSM to capture spatial-temporal
    inconsistencies that AI video generators produce. This is a purpose-built
    architecture for video authenticity detection, not a repurposed classifier.
    """

    BACKBONE_MAP = {
        "clip_vitl14": "vit_large_patch14_clip_224.openai",
        "clip_vitl14_laion": "vit_large_patch14_clip_224.laion2b_s32b_b82k",
        "clip_vitb16": "vit_base_patch16_clip_224.openai",
    }

    ALLOWED_KWARGS = {
        "mamba_dim", "fusing_ratios", "mamba_d_state", "mamba_expand",
        "mamba_drop", "use_tta", "num_frames",
    }

    def __init__(
        self,
        num_classes=2,
        backbone="clip_vitl14",
        pretrained=True,
        mamba_dim=512,
        fusing_ratios=2,
        mamba_d_state=16,
        mamba_expand=2,
        mamba_drop=0.1,
        num_frames=8,
        use_tta=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_tta = use_tta
        self.mamba_dim = mamba_dim
        self.fusing_ratios = fusing_ratios
        self.num_frames = num_frames

        timm_name = self.BACKBONE_MAP.get(backbone, backbone)
        self.backbone = timm.create_model(
            timm_name, pretrained=pretrained,
            num_classes=0, global_pool="",
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.backbone.forward_features(dummy)
            self.backbone_dim = out.shape[-1]
            self.num_tokens = out.shape[1]
            self.grid_size = int(math.sqrt(self.num_tokens - 1))

        assert self.grid_size % fusing_ratios == 0, (
            f"grid_size {self.grid_size} not divisible by fusing_ratios {fusing_ratios}"
        )

        self.s = self.grid_size // fusing_ratios
        self.patch_nums = self.s * self.s

        self.proj = nn.Linear(self.backbone_dim, mamba_dim)

        self.register_buffer(
            "pixel_mean",
            torch.tensor(CLIP_MEAN).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(CLIP_STD).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        mamba_config = MambaConfig(
            d_model=mamba_dim,
            d_state=mamba_d_state,
            expand_factor=mamba_expand,
            drop_prob=mamba_drop,
            bimamba=True,
            use_pscan=True,
        )
        self.mamba = MambaResidualBlock(mamba_config)

        self.fc_norm = nn.LayerNorm(self.patch_nums * mamba_dim)
        self.fc_norm_global = nn.LayerNorm(mamba_dim)

        self.classifier = nn.Linear(
            (self.patch_nums + 1) * mamba_dim, num_classes
        )

        self._init_head_weights()

    def _init_head_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_backbone_except_ln(self):
        """Freeze backbone except LayerNorm params (LN-tuning strategy)."""
        self.freeze_backbone()
        ln_count = 0
        for name, param in self.backbone.named_parameters():
            if "norm" in name.lower() or "ln_" in name.lower():
                param.requires_grad = True
                ln_count += param.numel()
        return ln_count

    def _forward_single(self, x):
        """Core forward pass on normalized input [B, T, C, H, W]."""
        B, T, C, H, W = x.shape

        frames = x.reshape(B * T, C, H, W)
        features = self.backbone.forward_features(frames)
        cls_tokens = features[:, 0, :]
        patch_tokens = features[:, 1:, :]

        cls_tokens = self.proj(cls_tokens)
        patch_tokens = self.proj(patch_tokens)

        global_feat = cls_tokens.view(B, T, -1).mean(1)
        global_feat = self.fc_norm_global(global_feat)

        c = self.mamba_dim
        g = self.grid_size
        fr = self.fusing_ratios
        s = self.s

        patch_tokens = patch_tokens.view(B, T, g, g, c)
        patch_tokens = patch_tokens.view(B, T, fr, s, fr, s, c)
        patch_tokens = patch_tokens.permute(
            0, 2, 4, 1, 3, 5, 6
        ).contiguous()
        patch_tokens = patch_tokens.view(B * s * s, T, fr * fr, c)

        patch_tokens = _reorder_patches(patch_tokens, fr)

        patch_tokens = patch_tokens.permute(0, 2, 1, 3).contiguous()
        patch_tokens = patch_tokens.view(B * s * s, -1, c)

        mamba_out = self.mamba(patch_tokens)

        mamba_feat = mamba_out.mean(1)
        mamba_feat = mamba_feat.view(B, -1)
        mamba_feat = self.fc_norm(mamba_feat)

        combined = torch.cat([global_feat, mamba_feat], dim=1)
        return self.classifier(combined)

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] float32 with values 0-255 (from gasbench)
        Returns:
            logits: [B, num_classes] (always float32)
        """
        x = x.float() / 255.0
        pm = self.pixel_mean.float()
        ps = self.pixel_std.float()
        x = (x - pm) / ps

        use_amp = (not self.training) and (x.device.type == "cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            if self.use_tta:
                logits_orig = self._forward_single(x)
                logits_flip = self._forward_single(x.flip(-1))
                logits = (logits_orig + logits_flip) / 2
            else:
                logits = self._forward_single(x)

        return logits.float()

    def get_param_groups(self, base_lr, backbone_lr_scale=0.01,
                         weight_decay=0.05):
        """Parameter groups: separate backbone (low LR) from head (high LR)."""
        bb_wd, bb_nowd = [], []
        head_wd, head_nowd = [], []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            no_wd = name.endswith(".bias") or "norm" in name.lower()
            if name.startswith("backbone"):
                (bb_nowd if no_wd else bb_wd).append(param)
            else:
                (head_nowd if no_wd else head_wd).append(param)

        return [
            {"params": bb_wd, "lr": base_lr * backbone_lr_scale,
             "weight_decay": weight_decay, "name": "backbone_decay"},
            {"params": bb_nowd, "lr": base_lr * backbone_lr_scale,
             "weight_decay": 0.0, "name": "backbone_no_decay"},
            {"params": head_wd, "lr": base_lr,
             "weight_decay": weight_decay, "name": "head_decay"},
            {"params": head_nowd, "lr": base_lr,
             "weight_decay": 0.0, "name": "head_no_decay"},
        ]


def create_model(num_classes=2, backbone="clip_vitl14", pretrained=True,
                  **kwargs):
    filtered = {k: v for k, v in kwargs.items()
                if k in CLIPDeMambaDetector.ALLOWED_KWARGS}
    return CLIPDeMambaDetector(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        **filtered,
    )


# ============================================================================
# gasbench load_model interface (required by custom_model_loader.py)
# ============================================================================

def load_model(weights_path, num_classes=2, backbone="clip_vitl14", **kwargs):
    """Load trained model for gasbench evaluation.

    Called by gasbench custom_model_loader.py:
      model = module.load_model(weights_path, **model_config)
    """
    for drop_key in ("dtype", "bf16", "config", "weights_file"):
        kwargs.pop(drop_key, None)

    model = CLIPDeMambaDetector(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=False,
        **{k: v for k, v in kwargs.items()
           if k in CLIPDeMambaDetector.ALLOWED_KWARGS},
    )

    if str(weights_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(
            weights_path, map_location="cpu", weights_only=True
        )
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict, strict=True)
    model.train(False)
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Testing CLIPDeMambaDetector")
    print("=" * 60)

    model = create_model()

    total = sum(p.numel() for p in model.parameters())
    backbone_params = sum(
        p.numel() for n, p in model.named_parameters()
        if n.startswith("backbone")
    )
    head_params = total - backbone_params
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"Backbone:       CLIP ViT-L/14 @ 224x224")
    print(f"Backbone dim:   {model.backbone_dim}")
    print(f"Grid size:      {model.grid_size}x{model.grid_size} = "
          f"{model.grid_size ** 2} patches")
    print(f"Mamba dim:      {model.mamba_dim}")
    print(f"Fusing ratios:  {model.fusing_ratios}")
    print(f"Spatial groups: {model.patch_nums}")
    print(f"Total params:   {total / 1e6:.1f}M")
    print(f"  Backbone:     {backbone_params / 1e6:.1f}M")
    print(f"  Head:         {head_params / 1e6:.1f}M")
    print(f"All trainable:  {trainable / 1e6:.1f}M")

    ln_count = model.freeze_backbone_except_ln()
    ln_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"\nWith LN-tuning:")
    print(f"  LN params:    {ln_count:,}")
    print(f"  Trainable:    {ln_trainable / 1e6:.2f}M")

    model.unfreeze_backbone()

    print(f"\nForward pass test:")
    dummy = torch.randint(0, 256, (2, 8, 3, 224, 224), dtype=torch.float32)
    with torch.no_grad():
        logits = model(dummy)
    print(f"  Input:  {dummy.shape} (float32, 0-255)")
    print(f"  Output: {logits.shape}")
    print(f"  Logits: {logits}")
    print("OK")
