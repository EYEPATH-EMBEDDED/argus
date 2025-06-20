"""Geometry utilities – collision triangle, overlap ratio, weight value."""
from __future__ import annotations

import math
import logging
from typing import Tuple

from shapely.geometry import Polygon

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collision‑triangle 생성 (해상도 1280×720 기준)
# ---------------------------------------------------------------------------


def _build_collision_triangle(
    *,
    height_m: float = 1.70,
    torso_m: float = 0.45,
    theta_v_deg: float = 52.0,
    theta_h_deg: float = 65.0,
    img_w: int = 1280,
    img_h: int = 720,
) -> Polygon:
    """사람 키·시야각 → 화면 좌표상 삼각형(POLYGON) 반환."""
    _LOGGER.debug("Building collision triangle: h=%.2fm, torso=%.2fm", height_m, torso_m)

    d = height_m * math.tan(math.radians(theta_v_deg) / 2)  # 눈 ~ ground 거리
    base_ratio = torso_m / (2 * d * math.tan(math.radians(theta_h_deg) / 2))
    half_base = base_ratio * img_w / 2

    apex = (img_w / 2, img_h / 2)  # (640,360)
    left = (apex[0] - half_base, img_h)
    right = (apex[0] + half_base, img_h)
    return Polygon([apex, left, right])


# 전역 단일 인스턴스 (immutable)
_COLLISION_POLY: Polygon = _build_collision_triangle()
_COLLISION_AREA: float = float(_COLLISION_POLY.area)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def overlap_ratio(box_xyxy: Tuple[float, float, float, float]) -> float:
    """bbox(x1,y1,x2,y2) 대비 삼각형 overlap 면적 비율 [0,1]."""
    x1, y1, x2, y2 = box_xyxy
    if x2 <= x1 or y2 <= y1:  # 빈 박스 방지
        return 0.0

    bbox_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    inter_area = _COLLISION_POLY.intersection(bbox_poly).area
    ratio = inter_area / _COLLISION_AREA if _COLLISION_AREA else 0.0
    _LOGGER.debug("overlap_ratio: %.4f", ratio)
    return ratio


def weight_value(
    box_xyxy: Tuple[float, float, float, float], *, alpha: float = 5.0, beta: float = 5.0
) -> float:
    """가중치 예시 함수: x=640 & y=0 부근일수록 ↑."""
    x1, y1, x2, y2 = box_xyxy
    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
    half_base = (_COLLISION_POLY.bounds[2] - _COLLISION_POLY.bounds[0]) / 2

    x_rel = (xc - 640.0) / half_base
    wx = math.exp(-alpha * x_rel**2)
    wy = math.exp(-beta * (yc / 360.0) ** 2)
    return wx * wy