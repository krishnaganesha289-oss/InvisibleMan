import logging
import time

import cv2
import numpy as np

from config.settings import DEFAULT_CONFIG
from invisibility.background import BackgroundCapture
from invisibility.cloak import InvisibilityCloak
from invisibility.hand_gesture import ClaspGestureController, HandClaspDetector
from utils.camera import CameraManager

logger = logging.getLogger("invisible_man")


def _draw_lines(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    output = frame.copy()
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (12, 28 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def _capture_background(
    camera: CameraManager,
    capturer: BackgroundCapture,
    cloak: InvisibilityCloak,
    window_name: str,
) -> bool:
    deadline = time.monotonic() + DEFAULT_CONFIG.background_countdown_seconds
    while time.monotonic() < deadline:
        frame = camera.read()
        remaining = max(1, int(deadline - time.monotonic()) + 1)
        display = _draw_lines(
            frame,
            [
                f"Background capture in {remaining}s - leave the frame empty",
                "Q: quit",
            ],
        )
        cv2.imshow(window_name, display)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            return False

    def show_progress(frame: np.ndarray, current: int, total: int) -> bool:
        display = _draw_lines(
            frame,
            [f"Capturing background: {current}/{total}", "Q: quit"],
        )
        cv2.imshow(window_name, display)
        return (cv2.waitKey(1) & 0xFF) not in (ord("q"), ord("Q"))

    try:
        background = capturer.capture(camera.read, show_progress)
    except InterruptedError:
        return False
    cloak.set_background(background)
    return True


def run() -> int:
    config = DEFAULT_CONFIG
    camera = CameraManager(config.camera_index, config.camera_width, config.camera_height)
    cloak: InvisibilityCloak | None = None
    hand_detector: HandClaspDetector | None = None
    try:
        camera.open()
        cloak = InvisibilityCloak(config)
        hand_detector = HandClaspDetector(
            config.hand_landmarker_model_path,
            config.hand_clasp_distance_ratio,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        if hand_detector is not None:
            hand_detector.close()
        if cloak is not None:
            cloak.close()
        camera.release()
        return 1

    assert cloak is not None
    assert hand_detector is not None
    gesture_controller = ClaspGestureController(
        config.gesture_hold_seconds, config.minimum_invisible_seconds
    )
    capturer = BackgroundCapture(
        config.background_frame_count, config.background_frame_interval_seconds
    )
    notification = ""
    notification_time = 0.0

    try:
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
        if not _capture_background(camera, capturer, cloak, config.window_name):
            return 0

        while True:
            frame = camera.read()
            clasped = hand_detector.hands_are_clasped(frame)
            changed_state = gesture_controller.update(clasped)
            if changed_state is not None:
                notification = (
                    "Hands clasped - "
                    f"{'INVISIBLE' if changed_state else 'VISIBLE'}"
                )
                notification_time = time.monotonic()

            invisible = gesture_controller.invisible
            output = cloak.apply(frame, invisible)
            status = "INVISIBLE" if invisible else "VISIBLE"
            lines = [
                f"Status: {status}",
                "Clasp both hands for 0.6s to toggle | Q: quit | B: recapture background",
                (
                    f"Hands: {hand_detector.hands_detected}/2 detected; "
                    f"{'together' if hand_detector.clasped else 'bring them together'}"
                ),
            ]
            if invisible:
                wait_seconds = gesture_controller.seconds_until_reappear()
                lines.append(
                    f"Clasp again to reappear in {wait_seconds:.1f}s"
                    if wait_seconds > 0
                    else "Clasp both hands for 0.6s to reappear"
                )
            else:
                lines.append(
                    "Show both hands clearly; release between each gesture"
                )
            if notification and time.monotonic() - notification_time < 2.0:
                lines.append(notification)
            cv2.imshow(config.window_name, _draw_lines(output, lines))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("b"), ord("B")):
                if not _capture_background(camera, capturer, cloak, config.window_name):
                    break
    except (cv2.error, RuntimeError) as error:
        logger.error("Application error: %s", error)
        return 1
    finally:
        camera.release()
        hand_detector.close()
        cloak.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(run())