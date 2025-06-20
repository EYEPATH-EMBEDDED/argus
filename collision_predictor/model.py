import logging
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn as nn

from .queue import SlidingQueue
from .feature import build_feature
from .geometry import overlap_ratio

_LOGGER = logging.getLogger(__name__)


class ConvLSTMClassifier(nn.Module):
    """Conv1D → Bi‑LSTM → FC (binary classification)."""

    def __init__(
        self,
        input_dim: int = 7,
        conv_channels: int = 32,
        lstm_hidden: int = 64,
        num_classes: int = 2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, conv_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(conv_channels)
        self.act = nn.ReLU()
        self.lstm = nn.LSTM(
            conv_channels,
            lstm_hidden,
            batch_first=True,
            bidirectional=bidirectional,
        )
        d = lstm_hidden * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,F)
        x = self.act(self.bn(self.conv1d(x.permute(0, 2, 1))))  # (B,C,T)
        h, _ = self.lstm(x.permute(0, 2, 1))  # back to (B,T,C)
        return self.fc(h[:, -1])  # (B,2)


class CollisionPredictor:
    """End‑to‑end predictor: YOLO bbox → Conv‑LSTM prob."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        queue_len: int = 1200,
        device: str | torch.device | None = None,
        threshold: float = 0.55,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = float(threshold)
        self.queue = SlidingQueue(queue_len)

        self.t0: float | None = None

        self.model = ConvLSTMClassifier()
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt)
        self.model.eval().to(self.device)
        _LOGGER.info("Loaded Conv‑LSTM weight: %s", model_path)

    # -----------------------------------------------------
    def update(
        self, *, 
        bboxes_xyxy, img_w:int, img_h:int,
        min_len:int = 300          # 300 row(≈15초) 이상이면 예측
    ) -> float | None:

        if not bboxes_xyxy:
            return None

        # ── (1) 프레임 안의 **모든 bbox**를 queue 에 push ──────────
        for box in bboxes_xyxy:
            feat = build_feature(
                #frame_idx=frame_idx,
                #time_sec=time_sec,
                box_xyxy=box,
                img_w=img_w,
                img_h=img_h,
            )
            self.queue.append(feat.tolist())

        # ── (2) row 수가 min_len 이상이면 곧바로 예측 ─────────────
        if len(self.queue._dq) >= min_len:
            x = torch.from_numpy(self.queue.as_numpy()).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prob = torch.softmax(self.model(x), 1)[0, 1].item()
            return prob
        return None

    def is_danger(self, prob: float | None) -> bool:
        return prob is not None and prob >= self.threshold
