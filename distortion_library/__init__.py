"""
Distortion Library
==================
Differentiable image augmentation for computer vision.
"""

from .blur_atmospheric_turbulence_v1 import blur_atmospheric_turbulence_v1
from .blur_frc_optical_flow_v1 import blur_frc_optical_flow_v1
from .compression_recompression_v1 import (
    distortion as compression_recompression_v1,
)
from .mgtc_cascade_v1 import mgtc_cascade_v1

try:
    from .motion_blur_linear import DISTORTION_REGISTRY
except ImportError:
    from .blur_atmospheric_turbulence_v1 import DISTORTION_REGISTRY

__version__ = "0.4.0"

__all__ = [
    "DISTORTION_REGISTRY",
    "blur_atmospheric_turbulence_v1",
    "blur_frc_optical_flow_v1",
    "compression_recompression_v1",
    "mgtc_cascade_v1",
]
