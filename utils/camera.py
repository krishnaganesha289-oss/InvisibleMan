import cv2
import numpy as np


class CameraManager:
    def __init__(
        self, camera_index: int = 0, width: int = 640, height: int = 480
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self._capture is not None and self._capture.isOpened():
            return

        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not open webcam at index {self.camera_index}. "
                "Check that it is connected and not being used by another application."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture = capture

    def read(self) -> np.ndarray:
        if self._capture is None or not self._capture.isOpened():
            raise RuntimeError("Webcam is not open")
        success, frame = self._capture.read()
        if not success or frame is None or frame.size == 0:
            raise RuntimeError("Could not read a frame from the webcam")
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "CameraManager":
        self.open()
        return self

    def __exit__(
        self, _exc_type: object, _exc_value: object, _traceback: object
    ) -> None:
        del _exc_type, _exc_value, _traceback
        self.release()