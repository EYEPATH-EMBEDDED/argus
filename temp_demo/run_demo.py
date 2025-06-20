"""Video -> annotated video demo."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2  # type: ignore

from collision_predictor import CollisionPredictor
from collision_predictor.yolo_utils import detect_bboxes

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
_LOGGER = logging.getLogger("demo")


def run_demo(video_path: str | Path, output_path: str | Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    predictor = CollisionPredictor(
        model_path="assets/first_best_model.pth", threshold=0.01
    )

    frame_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bboxes = detect_bboxes(frame, weights_path="assets/last11.pt")
        prob = predictor.update(
            frame_idx=frame_idx,
            time_sec = frame_idx / fps,
            bboxes_xyxy=bboxes,
            img_w=w,              
            img_h=h,              
        )

        if predictor.is_danger(prob):
            cv2.putText(
                frame,
                f"DANGER {prob:.2f}",
                (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 0, 255),
                6,
            )

        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )

        out.write(frame)

        # 👇 실시간 화면 출력 추가
        frame_idx += 1

    cap.release()
    out.release()
    fps_total = frame_idx / (time.time() - t0)
    _LOGGER.info("Finished. Saved → %s (%.1f fps)", output_path, fps_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="input video file path")
    parser.add_argument("-o", "--output", default="demo/output.mp4")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run_demo(args.video, args.output)
