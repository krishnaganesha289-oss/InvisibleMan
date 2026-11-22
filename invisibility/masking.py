import cv2
import numpy as np


def create_person_mask(
    person_probability: np.ndarray,
    threshold: float = 0.5,
    kernel_size: int = 5,
    blur_size: int = 9,
) -> np.ndarray:
    """Convert person confidence values into a cleaned, soft alpha mask."""
    if person_probability.ndim != 2:
        raise ValueError("Expected a single-channel person confidence mask")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")
    if blur_size < 1 or blur_size % 2 == 0:
        raise ValueError("blur_size must be a positive odd number")

    mask = (person_probability >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.GaussianBlur(mask, (blur_size, blur_size), 0)


def composite_background(
    frame: np.ndarray, background: np.ndarray, alpha_mask: np.ndarray
) -> np.ndarray:
    """Blend the background into masked pixels while retaining the live image elsewhere."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR image")
    if background.ndim != 3 or background.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR background")
    if alpha_mask.shape != frame.shape[:2]:
        raise ValueError("Mask dimensions must match the frame")

    if background.shape[:2] != frame.shape[:2]:
        background = cv2.resize(
            background, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
        )

    alpha = alpha_mask.astype(np.float32)[..., np.newaxis] / 255.0
    blended = frame.astype(np.float32) * (1.0 - alpha)
    blended += background.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)