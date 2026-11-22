
import numpy as np

from config.settings import AppConfig, DEFAULT_CONFIG
from invisibility.masking import composite_background, create_person_mask
from invisibility.person_segmentation import MediaPipePersonSegmenter, PersonMasker


class InvisibilityCloak:
    def __init__(
        self,
        config: AppConfig = DEFAULT_CONFIG,
        segmenter: PersonMasker | None = None,
    ) -> None:
        self._config = config
        self._segmenter = (
            segmenter
            if segmenter is not None
            else MediaPipePersonSegmenter(config.segmentation_model_path)
        )
        self._kernel_size = config.mask_kernel_size
        self._blur_size = config.mask_blur_size
        self._background: np.ndarray | None = None

    def set_background(self, background: np.ndarray) -> None:
        if background.ndim != 3 or background.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR background")
        self._background = background.copy()

    def mask(self, frame: np.ndarray) -> np.ndarray:
        person_probability = self._segmenter.segment(frame)
        return create_person_mask(
            person_probability,
            self._config.person_probability_threshold,
            self._kernel_size,
            self._blur_size,
        )

    def apply(self, frame: np.ndarray, invisible: bool) -> np.ndarray:
        if not invisible:
            return frame
        if self._background is None:
            raise RuntimeError("Background has not been captured")

        person_mask = self.mask(frame)
        return composite_background(frame, self._background, person_mask)

    def close(self) -> None:
        self._segmenter.close()