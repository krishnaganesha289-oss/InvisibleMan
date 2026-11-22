# Invisible Man – Real-Time Computer Vision Invisibility System

A fully local Python application that segments a person in a live webcam feed
and replaces them with the captured background. Clasp both hands to toggle the
effect; clothing color and microphone access do not matter.

## Requirements

- Python 3.10 or newer
- A webcam
- A desktop session that supports OpenCV windows
- The bundled MediaPipe selfie-segmentation and hand-landmark models in `models/`

OpenCV, MediaPipe, and NumPy are installed from the package index. Both models
run locally; the application makes no network requests while running and uses
no paid or cloud services.

## Setup

From the project directory, create and activate a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On systems where PowerShell blocks activation, run
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and use
`.\.venv\Scripts\python.exe main.py` instead.

## Run

```powershell
python main.py
```

The app opens the default webcam and shows a five-second countdown. Leave the
camera view empty during the countdown and background capture. Move into view
wearing ordinary clothing and keep both hands visible to the camera.

- Clasp both hands together for about 0.6 seconds to toggle visibility.
- After becoming invisible, stay invisible for at least 10 seconds. Release your
  hands, then clasp them together again for about 0.6 seconds to reappear.
- Release your hands between gestures; holding them together does not toggle
  repeatedly.
- Press **B** to recapture the empty background (including the countdown).
- Press **Q** to quit.

Gesture hold duration, minimum invisible duration, clasp-distance threshold,
person confidence threshold, camera size, and background capture settings are
in `config/settings.py`.

The bundled `selfie_segmenter.tflite` model is the MediaPipe Selfie Segmenter
model distributed by Google:
<https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite>.
The bundled `hand_landmarker.task` model is the MediaPipe Hand Landmarker model:
<https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task>.

## Tests

Run the hardware-independent tests from the project directory:

```powershell
python -m unittest discover -s tests -v
```

The tests cover person-mask compositing, bundled model inference, clasp gesture
debouncing and timeout behavior, and stable median background capture. They do
not require a webcam or microphone.

## Limitations

- The camera must remain fixed between background capture and use. Recapture
  after moving it or changing the scene.
- Person segmentation can have imperfect edges, especially with motion,
  low light, occlusion, or unusual poses. It does not depend on clothing color.
- Clasp detection requires both hands to be visible and can miss gestures in
  poor lighting, with occlusion, or at unusual camera angles.
- Camera selection uses the system default (camera index 0).
- Gesture response and segmentation quality with a person in frame should be
  verified manually on the target machine.