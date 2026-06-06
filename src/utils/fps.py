"""
FPS (Frames Per Second) calculation utility.

Uses a rolling window approach for stable, accurate FPS measurement.
"""

import time
from collections import deque
from typing import Deque


class FPSCounter:
    """
    Rolling-window FPS counter.

    Maintains a sliding window of frame timestamps to compute
    a smoothed FPS value without per-frame oscillation.

    Args:
        window_size: Number of recent frames to include in the average.
    """

    def __init__(self, window_size: int = 30) -> None:
        self._window_size: int = window_size
        self._timestamps: Deque[float] = deque(maxlen=window_size)
        self._fps: float = 0.0

    def tick(self) -> float:
        """
        Record a new frame timestamp and return the current FPS.

        Returns:
            Current frames-per-second estimate.
        """
        now = time.monotonic()
        self._timestamps.append(now)

        if len(self._timestamps) >= 2:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0:
                self._fps = (len(self._timestamps) - 1) / elapsed

        return self._fps

    @property
    def fps(self) -> float:
        """Last computed FPS value."""
        return self._fps

    def reset(self) -> None:
        """Clear all recorded timestamps."""
        self._timestamps.clear()
        self._fps = 0.0
