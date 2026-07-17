"""
landmarks.py
============
Hand landmark extraction using MediaPipe's modern Tasks API
(mp.tasks.vision.HandLandmarker).

The older mp.solutions.hands API has been dropped from recent MediaPipe
wheels (0.10.30+) on some platforms -- mediapipe.python doesn't even exist
in those builds. This uses the currently-supported replacement instead.

extract_landmarks(image) -> np.ndarray of shape (42,) or None
    21 landmarks * (x, y), normalized relative to the wrist and hand size
    so the feature vector is invariant to hand position/scale in the frame.
    Returns None if no hand is detected.

On first import, this downloads a small (~7MB) model file
(hand_landmarker.task) into the same folder as this script, if not already
present.
"""

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODEL_FILENAME = "hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _MODEL_FILENAME)


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print(f"Downloading hand landmark model to {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")


_ensure_model()

_base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
_options = mp_vision.HandLandmarkerOptions(
    base_options=_base_options,
    num_hands=1,
    running_mode=mp_vision.RunningMode.IMAGE,
)
_detector = mp_vision.HandLandmarker.create_from_options(_options)


def extract_landmarks(image: np.ndarray):
    """
    Extract a normalized, flattened hand-landmark feature vector from an image.

    Args:
        image: BGR image (as read by cv2.imread / cv2.VideoCapture)

    Returns:
        np.ndarray of shape (42,) -- [x0, y0, x1, y1, ..., x20, y20] -- or None
        if no hand was detected.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    result = _detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]
    coords = np.array([[lm.x, lm.y] for lm in hand], dtype=np.float32)  # (21, 2)

    # Normalize: translate so wrist (landmark 0) is the origin, then scale by
    # the wrist-to-middle-fingertip distance so hand size/distance from camera
    # doesn't affect the features.
    wrist = coords[0]
    coords = coords - wrist

    middle_tip = coords[12]
    scale = np.linalg.norm(middle_tip)
    if scale < 1e-6:
        scale = 1.0
    coords = coords / scale

    return coords.flatten()  # (42,)


def draw_landmarks(image: np.ndarray):
    """
    Optional helper: draw hand landmarks on an image for debugging/visualization.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = _detector.detect(mp_image)

    annotated = image.copy()
    h, w = annotated.shape[:2]

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in connections:
            cv2.line(annotated, pts[a], pts[b], (0, 255, 0), 2)
        for i, (x, y) in enumerate(pts):
            color = (0, 0, 255) if i == 0 else (255, 0, 0)
            cv2.circle(annotated, (x, y), 4, color, -1)

    return annotated