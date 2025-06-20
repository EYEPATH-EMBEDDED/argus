"""Fixed‑length sliding window with optional zero‑padding."""
from __future__ import annotations

import numpy as np
from collections import deque
from typing import Deque, List


class SlidingQueue:
    """큐 길이가 부족하면 앞쪽을 0으로 패딩한다."""

    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self._dq: Deque[List[float]] = deque(maxlen=maxlen)

    # ---------------------------------------------------------------------
    # Queue operations
    # ---------------------------------------------------------------------

    def append(self, vec: List[float]) -> None:
        if len(vec) == 0:
            raise ValueError("feature vector must not be empty")
        self._dq.append(vec)

    def is_full(self) -> bool:  # noqa: D401
        """True if len == maxlen"""
        return len(self._dq) == self.maxlen

    def as_numpy(self) -> np.ndarray:  # shape (maxlen, F)
        if not self._dq:
            raise RuntimeError("queue is empty")

        arr = np.array(self._dq, dtype=np.float32)
        if len(arr) < self.maxlen:
            pad = np.zeros((self.maxlen - len(arr), arr.shape[1]), dtype=np.float32)
            arr = np.vstack([pad, arr])
        return arr