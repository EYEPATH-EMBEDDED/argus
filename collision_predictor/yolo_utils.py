"""YOLOv8 wrapper – singleton pattern for model reuse."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

import cv2  # type: ignore
from ultralytics import YOLO  # type: ignore

_LOGGER = logging.getLogger(__name__)
_MODEL_CACHE = {}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def detect_bboxes(
    img_bgr: "cv2.Mat",
    *,
    weights_path: str | Path,
    conf: float = 0.25,
    iou: float = 0.45,
    max_boxes: int = 30,
) -> List[List[float]]:
    """단일 이미지 → [[x1,y1,x2,y2], ...]. 좌표는 float."""
    path = str(weights_path)
    model = _get_model(path, conf=conf, iou=iou)
    _LOGGER.debug("Running YOLO inference (conf=%.2f, iou=%.2f)", conf, iou)

    try:
        results = model(img_bgr, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        return boxes[: max_boxes].tolist()
    except Exception as exc:  # pylint: disable=broad-except
        _LOGGER.exception("YOLO inference failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_model(path: str, *, conf: float, iou: float) -> YOLO:
    if path not in _MODEL_CACHE:
        _LOGGER.info("Loading YOLO weights from %s", path)
        model = YOLO(path)
        model.conf, model.iou = conf, iou
        _MODEL_CACHE[path] = model
    return _MODEL_CACHE[path]