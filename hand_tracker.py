"""
hand_tracker.py
---------------
Handles all MediaPipe hand detection, landmark extraction, and drawing utilities.

Uses the MediaPipe Tasks API (mp.tasks.python.vision.HandLandmarker), which is
the ONLY API available on mediapipe >= 0.10.21+ / Python 3.12+ / Apple Silicon.

All drawing is done with pure OpenCV — zero dependency on mp.solutions,
mp.framework.formats, or any other mediapipe sub-module that no longer ships
in the latest builds.
"""

import os
import urllib.request
import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ── Model download ───────────────────────────────────────────────────────────
MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_model():
    """
    Download hand_landmarker.task (~8 MB) once into the project folder.
    MediaPipe Tasks needs an explicit model bundle — it is NOT bundled inside
    the pip package anymore.
    """
    if not os.path.exists(MODEL_FILENAME):
        print(f"[INFO] Downloading MediaPipe hand model → {MODEL_FILENAME} …")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
            print("[INFO] Model downloaded successfully.")
        except Exception as e:
            print(f"[ERROR] Could not download model: {e}")
            print(f"        Download it manually from:\n        {MODEL_URL}")
            raise


# ── Hand skeleton connection pairs (MediaPipe spec, hard-coded) ──────────────
# Using a plain list avoids any dependency on mediapipe.solutions.hands
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (0, 17),
]


class HandTracker:
    """
    Real-time hand tracker using the MediaPipe Tasks HandLandmarker.

    Compatible with mediapipe >= 0.10 on ALL platforms (Apple M1/M2/M3/M4,
    Windows, Linux). Drawing uses only OpenCV — no mediapipe.solutions needed.

    Usage:
        tracker = HandTracker()
        all_landmarks = tracker.find_hands(frame)
        tracker.close()
    """

    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        """
        Initialise the HandLandmarker.

        Args:
            max_hands (int):               Maximum hands to detect (1 or 2).
            detection_confidence (float):  Min confidence for palm detection.
            tracking_confidence (float):   Min confidence to keep tracking.
        """
        _ensure_model()

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_FILENAME),
            running_mode=mp_vision.RunningMode.IMAGE,   # one detection per frame
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

    # ── Core public method ───────────────────────────────────────────────────

    def find_hands(self, frame, draw=True):
        """
        Detect hands in a BGR OpenCV frame.

        Args:
            frame (np.ndarray): BGR image from cv2.VideoCapture.
            draw  (bool):       Draw skeleton on the frame in-place when True.

        Returns:
            list[list[tuple]]: One list per detected hand, each containing
                               21 (x, y, z) normalised (0-1) tuples.
                               Returns [] when no hands are found.
        """
        # Tasks API requires an mp.Image in SRGB format
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)

        all_hands = []
        if result and result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_lms]
                all_hands.append(landmarks)
                if draw:
                    self._draw_hand(frame, landmarks)

        return all_hands

    # ── Pure-OpenCV drawing ──────────────────────────────────────────────────

    def _draw_hand(self, frame, landmarks):
        """
        Draw the 21 landmark dots and skeleton connections using cv2 only.

        Visual style:
          - Connections  → yellow lines
          - Regular dots → cyan filled circles
          - Fingertips (4, 8, 12, 16, 20) → larger magenta dots

        Args:
            frame     (np.ndarray): BGR image modified in-place.
            landmarks (list):       21 (x, y, z) normalised tuples.
        """
        h, w = frame.shape[:2]
        # Convert all normalised coords to pixel positions up-front
        pts = [(int(x * w), int(y * h)) for (x, y, _) in landmarks]

        # Draw bone connections underneath the dots
        for (a, b) in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 220, 255), 2, cv2.LINE_AA)

        # Draw landmark dots on top
        fingertips = {4, 8, 12, 16, 20}
        for idx, pt in enumerate(pts):
            if idx in fingertips:
                # Fingertips: larger magenta + white outline
                cv2.circle(frame, pt, 7, (255, 0, 200), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                # Regular joints: small cyan dot
                cv2.circle(frame, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)

    # ── Coordinate utilities ─────────────────────────────────────────────────

    def get_pixel_landmarks(self, landmarks, frame_shape):
        """
        Convert normalised (0-1) landmarks to integer pixel coordinates.

        Args:
            landmarks   (list):  21 (x, y, z) normalised tuples.
            frame_shape (tuple): Frame (height, width, channels).

        Returns:
            list[tuple[int,int]]: 21 (px, py) pixel pairs.
        """
        h, w = frame_shape[:2]
        return [(int(x * w), int(y * h)) for (x, y, _) in landmarks]

    def get_hand_center(self, pixel_landmarks):
        """
        Centroid of all 21 landmark pixel positions.

        Args:
            pixel_landmarks (list): (px, py) integer pairs.

        Returns:
            tuple[int, int]: (cx, cy) centroid.
        """
        xs = [p[0] for p in pixel_landmarks]
        ys = [p[1] for p in pixel_landmarks]
        return (int(np.mean(xs)), int(np.mean(ys)))

    def close(self):
        """Release MediaPipe detector resources."""
        self.detector.close()


# ──────────────────────────────────────────────
# MediaPipe Hand Landmark Index Reference
# ──────────────────────────────────────────────
# 0  = WRIST
# 1  = THUMB_CMC    2  = THUMB_MCP    3  = THUMB_IP     4  = THUMB_TIP
# 5  = INDEX_MCP    6  = INDEX_PIP    7  = INDEX_DIP     8  = INDEX_TIP
# 9  = MIDDLE_MCP   10 = MIDDLE_PIP   11 = MIDDLE_DIP    12 = MIDDLE_TIP
# 13 = RING_MCP     14 = RING_PIP     15 = RING_DIP      16 = RING_TIP
# 17 = PINKY_MCP    18 = PINKY_PIP    19 = PINKY_DIP     20 = PINKY_TIP