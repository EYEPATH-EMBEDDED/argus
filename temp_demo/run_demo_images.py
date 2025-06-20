#!/usr/bin/env python3
"""Images → annotated-video demo (fps 고정 20 Hz)."""
from __future__ import annotations

import argparse, logging, time
from pathlib import Path
import cv2                       # type: ignore

from collision_predictor import CollisionPredictor
from collision_predictor.yolo_utils import detect_bboxes

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(name)s: %(message)s')
_LOGGER = logging.getLogger("demo_images")


# ────────────────────────────────────────────────────────────
# util: 이미지 목록 수집 ― numeric sort  (1_0.jpg, 1_1.jpg …)
# ────────────────────────────────────────────────────────────
def _collect_image_paths(folder: Path,
                         exts=(".jpg", ".jpeg", ".png")) -> list[Path]:
    """
    • 파일명 패턴   :  <videoId>_<frameIdx>.<ext>   (예:  1_42.jpg)
    • 같은 videoId 안에서  frameIdx 숫자값으로 오름차순 정렬
    • videoId 가 여러 개면, 가장 첫 videoId 의 시퀀스만 사용
      (필요하면 CLI 인자로 선택하도록 쉽게 바꿀 수 있음)
    """
    candidates = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    parsed: dict[str, list[tuple[int, Path]]] = {}
    for p in candidates:
        try:
            vid, idx = p.stem.split("_", 1)
            idx_num = int(idx)
        except (ValueError, IndexError):
            _LOGGER.warning("Skip (unexpected name): %s", p.name)
            continue
        parsed.setdefault(vid, []).append((idx_num, p))

    if not parsed:
        raise FileNotFoundError(f"No valid <id>_<idx>.jpg images in {folder}")

    # 하나의 videoId(가장 앞에 오는 것)만 사용
    vid_selected = sorted(parsed.keys())[0]
    frames = sorted(parsed[vid_selected], key=lambda t: t[0])  # by idx_num
    _LOGGER.info("Use sequence id=%s  |  %d frames found", vid_selected, len(frames))
    return [p for _, p in frames]


# ────────────────────────────────────────────────────────────
# main demo
# ────────────────────────────────────────────────────────────
def run_demo_images(img_dir: str | Path, output_path: str | Path,
                    *, fps: int = 20) -> None:

    img_paths = _collect_image_paths(Path(img_dir))

    # 첫 장으로 해상도 결정
    first = cv2.imread(str(img_paths[0]))
    if first is None:
        raise RuntimeError(f"Cannot read image: {img_paths[0]}")
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    predictor = CollisionPredictor(
        model_path="assets/first_best_model.pth",
        threshold=0.01,              # ← run_demo.py 와 동일
    )

    t0 = time.time()
    for frame_idx, path in enumerate(img_paths):
        frame = cv2.imread(str(path))
        if frame is None:
            _LOGGER.warning("Skip unreadable image: %s", path)
            continue

        # YOLO + Conv-LSTM 업데이트
        bboxes = detect_bboxes(frame, weights_path="assets/last11.pt")
        prob = predictor.update(
            frame_idx=frame_idx,              # ← 정수 idx 그대로
            time_sec=frame_idx / fps,
            bboxes_xyxy=bboxes,
            img_w=w, img_h=h,
        )

        # 오버레이
        if predictor.is_danger(prob):
            cv2.putText(frame, f"DANGER {prob:.2f}", (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run collision demo on an image sequence.")
    ap.add_argument("image_dir", help="directory that contains frame images")
    ap.add_argument("-o", "--output", default="demo/output_images.mp4",
                    help="output MP4 path")
    ap.add_argument("--fps", type=int, default=20,
                    help="frame-rate for the output video (default 20)")
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run_demo_images(args.image_dir, args.output, fps=args.fps)
