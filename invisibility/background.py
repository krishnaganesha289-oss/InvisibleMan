import time
from collections.abc import Callable

import numpy as np


FrameReader = Callable[[], np.ndarray]
ProgressCallback = Callable[[np.ndarray, int, int], bool | None]


class BackgroundCapture:
    """Build a stable background from multiple frames captured from a fixed camera."""

    def __init__(self, frame_count: int = 30, interval_seconds: float = 0.05) -> None:
        if frame_count < 1:
            raise ValueError("frame_count must be at least one")
        if interval_seconds < 0:
            raise ValueError("interval_seconds cannot be negative")
        self.frame_count = frame_count
        self.interval_seconds = interval_seconds

    def capture(
        self,
        read_frame: FrameReader,
        on_progress: ProgressCallback | None = None,
    ) -> np.ndarray:
        frames: list[np.ndarray] = []
        expected_shape: tuple[int, ...] | None = None

        for index in range(self.frame_count):
            frame = read_frame()
            if frame is None or frame.size == 0:
                raise RuntimeError("The camera returned an empty frame during capture")
            if expected_shape is None:
                expected_shape = frame.shape
            elif frame.shape != expected_shape:
                raise RuntimeError("Camera frame dimensions changed during background capture")
            frames.append(frame.copy())

            if on_progress is not None and on_progress(frame, index + 1, self.frame_count) is False:
                raise InterruptedError("Background capture cancelled")
            if self.interval_seconds:
                time.sleep(self.interval_seconds)

        return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)