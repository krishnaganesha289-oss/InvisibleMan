import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


class HandClaspDetector:
    def __init__(
        self,
        model_path: Path,
        clasp_distance_ratio: float = 0.18,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Hand landmark model was not found: {model_path}")
        if not 0 < clasp_distance_ratio <= 1:
            raise ValueError("clasp_distance_ratio must be between zero and one")

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        try:
            self._detector = mp.tasks.vision.HandLandmarker.create_from_options(
                options
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not load hand landmark model: {model_path}"
            ) from error
        self._clasp_distance_ratio = clasp_distance_ratio
        self._last_timestamp_ms = 0
        self._hands_detected = 0
        self.clasped = False

    @property
    def hands_detected(self) -> int:
        return self._hands_detected

    def hands_are_clasped(self, frame: np.ndarray) -> bool:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR image")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = max(
            time.monotonic_ns() // 1_000_000, self._last_timestamp_ms + 1
        )
        self._last_timestamp_ms = timestamp_ms

        try:
            result = self._detector.detect_for_video(image, timestamp_ms)
        except Exception as error:
            raise RuntimeError("Hand landmark detection failed") from error
        self._hands_detected = len(result.hand_landmarks)
        if len(result.hand_landmarks) < 2:
            self.clasped = False
            return False

        palm_centers: list[np.ndarray] = []
        for landmarks in result.hand_landmarks[:2]:
            palm_points = np.asarray(
                [
                    (landmarks[index].x, landmarks[index].y)
                    for index in (0, 5, 9, 13, 17)
                ],
                dtype=np.float32,
            )
            palm_centers.append(palm_points.mean(axis=0))

        height, width = frame.shape[:2]
        scale = np.asarray((width, height), dtype=np.float32)
        distance = float(np.linalg.norm((palm_centers[0] - palm_centers[1]) * scale))
        frame_diagonal = float(np.hypot(width, height))
        self.clasped = distance / frame_diagonal <= self._clasp_distance_ratio
        return self.clasped

    def close(self) -> None:
        self._detector.close()


class ClaspGestureController:
    """Toggle visibility on one held clasp, with a minimum invisible duration."""

    def __init__(
        self,
        hold_seconds: float = 0.6,
        minimum_invisible_seconds: float = 10.0,
    ) -> None:
        if hold_seconds < 0:
            raise ValueError("hold_seconds cannot be negative")
        if minimum_invisible_seconds < 0:
            raise ValueError("minimum_invisible_seconds cannot be negative")
        self.hold_seconds = hold_seconds
        self.minimum_invisible_seconds = minimum_invisible_seconds
        self.invisible = False
        self._invisible_since: float | None = None
        self._clasp_started_at: float | None = None
        self._clasp_eligible = False
        self._gesture_consumed = False

    def update(self, clasped: bool, now: float | None = None) -> bool | None:
        timestamp = time.monotonic() if now is None else now
        if not clasped:
            self._clasp_started_at = None
            self._clasp_eligible = False
            self._gesture_consumed = False
            return None

        if self._clasp_started_at is None:
            self._clasp_started_at = timestamp
            self._clasp_eligible = (
                not self.invisible
                or self.seconds_until_reappear(timestamp) == 0
            )
        if self._gesture_consumed:
            return None
        if timestamp - self._clasp_started_at < self.hold_seconds:
            return None

        self._gesture_consumed = True
        if not self._clasp_eligible:
            return None

        self.invisible = not self.invisible
        self._invisible_since = timestamp if self.invisible else None
        return self.invisible

    def seconds_until_reappear(self, now: float | None = None) -> float:
        if not self.invisible or self._invisible_since is None:
            return 0.0
        timestamp = time.monotonic() if now is None else now
        return max(
            0.0,
            self.minimum_invisible_seconds - (timestamp - self._invisible_since),
        )
