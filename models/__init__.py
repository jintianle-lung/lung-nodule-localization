"""
Lung Nodule Localization Models
===============================

This package contains model definitions for the three-stage lung nodule localization system:

- Stage 1: Detection (DualStreamMSTCNDetector)
- Stage 2: Size Estimation (RawPositiveSizeModelV2)
- Stage 3: Depth Estimation (HierarchicalSharedWindowMTL)

Usage:
    from models import DualStreamMSTCNDetector
    from models import RawPositiveSizeModelV2
    from models import HierarchicalSharedWindowMTL
"""

from .dual_stream_mstcn_detection import (
    DualStreamMSTCNDetector,
    DualStreamMSTCNContextDetector,
    FrameEncoder2D,
    MultiScaleTemporalBlock,
    TemporalAttentionPooling,
)
from .task_protocol_v1 import (
    PROTOCOL_V1,
    SIZE_VALUES_CM,
    DEPTH_VALUES_CM,
    COARSE_DEPTH_ORDER,
    INPUT_SHAPE,
    protocol_summary,
)
from .input_normalization_v1 import (
    normalize_raw_frames_global,
    normalize_raw_frames_window_minmax,
    convert_sensor_to_pressure_maps,
)

__all__ = [
    "DualStreamMSTCNDetector",
    "DualStreamMSTCNContextDetector",
    "FrameEncoder2D",
    "MultiScaleTemporalBlock",
    "TemporalAttentionPooling",
    "PROTOCOL_V1",
    "SIZE_VALUES_CM",
    "DEPTH_VALUES_CM",
    "COARSE_DEPTH_ORDER",
    "INPUT_SHAPE",
    "protocol_summary",
    "normalize_raw_frames_global",
    "normalize_raw_frames_window_minmax",
    "convert_sensor_to_pressure_maps",
]
