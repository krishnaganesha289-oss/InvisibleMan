import unittest

import numpy as np

from config.settings import DEFAULT_CONFIG
from invisibility.hand_gesture import ClaspGestureController, HandClaspDetector


class ClaspGestureControllerTests(unittest.TestCase):
    def test_clasp_toggles_once_and_requires_release(self) -> None:
        controller = ClaspGestureController(hold_seconds=0.5)

        self.assertIsNone(controller.update(True, now=0.0))
        self.assertTrue(controller.update(True, now=0.5))
        self.assertIsNone(controller.update(True, now=1.0))
        self.assertIsNone(controller.update(False, now=1.1))
        self.assertIsNone(controller.update(True, now=2.0))
        self.assertFalse(controller.update(True, now=2.5))

    def test_cannot_reappear_until_invisible_for_ten_seconds(self) -> None:
        controller = ClaspGestureController(
            hold_seconds=0.5, minimum_invisible_seconds=10.0
        )
        controller.update(True, now=0.0)
        self.assertTrue(controller.update(True, now=0.5))
        controller.update(False, now=0.6)

        self.assertIsNone(controller.update(True, now=5.0))
        self.assertIsNone(controller.update(True, now=5.5))
        controller.update(False, now=5.6)
        self.assertIsNone(controller.update(True, now=10.4))
        self.assertIsNone(controller.update(True, now=10.9))
        controller.update(False, now=11.0)
        self.assertIsNone(controller.update(True, now=11.1))
        self.assertFalse(controller.update(True, now=11.6))

    def test_bundled_hand_model_runs_locally(self) -> None:
        detector = HandClaspDetector(DEFAULT_CONFIG.hand_landmarker_model_path)
        try:
            clasped = detector.hands_are_clasped(
                np.zeros((240, 320, 3), dtype=np.uint8)
            )
        finally:
            detector.close()

        self.assertFalse(clasped)


if __name__ == "__main__":
    unittest.main()
