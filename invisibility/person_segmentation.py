import time
from pathlib import Path
from typing import Protocol

import cv2
import mediapipe as mp
import numpy as np


class PersonMasker(Protocol):
    def segment(self, frame: np.ndarray) -> np.ndarray: ...

    def close(self) -> None: ...


class MediaPipePersonSegmenter:
    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Person segmentation model was not found: {model_path}"
            )

        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_category_mask=False,
            output_confidence_masks=True,
        )
        try:
            self._segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(
                options
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not load person segmentation model: {model_path}"
            ) from error
        self._last_timestamp_ms = 0

    def segment(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR image")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = max(
            time.monotonic_ns() // 1_000_000, self._last_timestamp_ms + 1
        )
        self._last_timestamp_ms = timestamp_ms
        result = self._segmenter.segment_for_video(image, timestamp_ms)
        if not result.confidence_masks:
            raise RuntimeError("Person segmentation returned no confidence masks")

        person_probability = np.squeeze(
            result.confidence_masks[0].numpy_view()
        ).copy()
        if person_probability.ndim != 2:
            raise RuntimeError("Person segmentation returned an invalid mask shape")
        if person_probability.shape != frame.shape[:2]:
            person_probability = cv2.resize(
                person_probability,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return person_probability

    def close(self) -> None:
        self._segmenter.close()
