# collision_predictor/feature.py
from __future__ import annotations
from typing import Sequence
import numpy as np
from .geometry import overlap_ratio

def build_feature(
    *, #frame_idx:int, time_sec:float,       # ← 인자는 그대로 두되
    box_xyxy:Sequence[float], img_w:int, img_h:int
) -> np.ndarray:
    """
    학습 단계에서 쓰인 7-D feature (frame_norm, time_norm 은 0) 로 변환
    """
    x1, y1, x2, y2 = box_xyxy
    vec = [
        0.0, 0.0,                   # frame_norm, time_norm → 항상 0
        x1 / img_w,
        y1 / img_h,
        x2 / img_w,
        y2 / img_h,
        overlap_ratio((x1, y1, x2, y2)),
    ]
    return np.asarray(vec, dtype=np.float32)
