from dataclasses import dataclass
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "selfie_segmenter.tflite"
HAND_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    background_countdown_seconds: int = 5
    background_frame_count: int = 30
    background_frame_interval_seconds: float = 0.05
    segmentation_model_path: Path = MODEL_PATH
    hand_landmarker_model_path: Path = HAND_MODEL_PATH
    person_probability_threshold: float = 0.5
    mask_kernel_size: int = 5
    mask_blur_size: int = 9
    hand_clasp_distance_ratio: float = 0.18
    gesture_hold_seconds: float = 0.6
    minimum_invisible_seconds: float = 10.0
    window_name: str = "Invisible Man - Invisibility System"


DEFAULT_CONFIG = AppConfig()