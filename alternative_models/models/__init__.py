"""
Video Models for Deepfake Detection

Models:
1. VideoMAE v2 ViT-L (primary, best generalization)
2. MViTv2-Base (optional ensemble companion)
3. Swin-L (2D backbone + temporal pooling)
4. Swin3D-L (true 3D Swin)
"""

from .video_mae_v2 import create_videomae_v2_vitl, VideoMAEv2ViTL
from .mvit_v2 import create_mvit_v2_base, MViTv2Base
from .swin_l import create_swin_l, SwinLargeVideo
from .swin3d_l import create_swin3d_l, Swin3DLarge
from .focal_loss import FocalLoss

__all__ = [
    'VideoMAEv2ViTL',
    'create_videomae_v2_vitl',
    'MViTv2Base',
    'create_mvit_v2_base',
    'SwinLargeVideo',
    'create_swin_l',
    'Swin3DLarge',
    'create_swin3d_l',
    'FocalLoss',
]

