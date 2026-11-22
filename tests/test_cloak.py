import unittest

import numpy as np

from config.settings import AppConfig
from invisibility.background import BackgroundCapture
from invisibility.cloak import InvisibilityCloak
from invisibility.masking import create_person_mask
from invisibility.person_segmentation import MediaPipePersonSegmenter


class FixedPersonSegmenter:
    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask

    def segment(self, frame: np.ndarray) -> np.ndarray:
        del frame
        return self._mask

    def close(self) -> None:
        pass


class CloakTests(unittest.TestCase):
    def test_person_is_replaced_regardless_of_clothing_color(self) -> None:
        config = AppConfig(mask_kernel_size=3, mask_blur_size=1)
        probability = np.zeros((21, 21), dtype=np.float32)
        probability[5:16, 5:16] = 1.0
        cloak = InvisibilityCloak(
            config, segmenter=FixedPersonSegmenter(probability)
        )
        background = np.full((21, 21, 3), (255, 0, 0), dtype=np.uint8)
        frame = np.zeros((21, 21, 3), dtype=np.uint8)
        frame[5:16, 5:16] = (0, 0, 255)
        cloak.set_background(background)

        result = cloak.apply(frame, invisible=True)

        np.testing.assert_array_equal(result[0, 0], frame[0, 0])
        np.testing.assert_array_equal(result[10, 10], background[10, 10])

    def test_visible_frame_is_returned_unchanged(self) -> None:
        frame = np.full((4, 4, 3), 42, dtype=np.uint8)
        cloak = InvisibilityCloak(segmenter=FixedPersonSegmenter(np.zeros((4, 4))))
        self.assertIs(cloak.apply(frame, invisible=False), frame)

    def test_person_confidence_becomes_a_mask(self) -> None:
        probability = np.zeros((9, 9), dtype=np.float32)
        probability[2:7, 2:7] = 0.9

        mask = create_person_mask(probability, threshold=0.5, kernel_size=1, blur_size=1)

        self.assertEqual(int(mask[4, 4]), 255)
        self.assertEqual(int(mask[0, 0]), 0)
        self.assertTrue(set(np.unique(mask)).issubset({0, 255}))

    def test_bundled_model_runs_locally(self) -> None:
        segmenter = MediaPipePersonSegmenter(AppConfig().segmentation_model_path)
        try:
            mask = segmenter.segment(np.zeros((64, 96, 3), dtype=np.uint8))
        finally:
            segmenter.close()

        self.assertEqual(mask.shape, (64, 96))
        self.assertTrue(np.isfinite(mask).all())

    def test_background_capture_uses_median_of_frames(self) -> None:
        values = iter((10, 100, 20))
        capture = BackgroundCapture(frame_count=3, interval_seconds=0)

        background = capture.capture(
            lambda: np.full((2, 2, 3), next(values), dtype=np.uint8)
        )

        np.testing.assert_array_equal(background, np.full((2, 2, 3), 20, np.uint8))

if __name__ == "__main__":
    unittest.main()