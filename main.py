"""
main.py
-------
Entry point for the JJK Hand Gesture Recognition System.

Runs the webcam loop, orchestrates hand tracking and gesture detection,
and renders all visual effects in real time.

Controls:
  ESC  → Exit
  'r'  → Reset effect state (if animation glitches)
"""

import cv2
import time

from hand_tracker import HandTracker
from gesture_utils import (
    detect_gesture,
    draw_gesture_effects,
    draw_gesture_label,
    draw_fps,
    draw_idle_hint,
    EffectState,
)


def main():
    # ── Webcam Setup ────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check that it is connected and not in use.")
        return

    # Set resolution (lower = faster, higher = more detail)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Webcam opened successfully.")
    print("[INFO] Press ESC to quit.")

    # ── Module Initialization ────────────────────────────────────────
    tracker      = HandTracker(max_hands=2, detection_confidence=0.7)
    effect_state = EffectState()   # One shared animation state

    # FPS tracking
    prev_time    = time.time()

    # Gesture smoothing — hold label for N frames to avoid flickering
    HOLD_FRAMES   = 8           # frames to keep a gesture label after it disappears
    gesture_label = None        # current displayed gesture
    gesture_hold  = 0           # countdown frames remaining

    # ── Main Loop ───────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from webcam.")
            break

        # Mirror the frame horizontally (selfie-mirror feel)
        frame = cv2.flip(frame, 1)

        # ── Hand Detection ───────────────────────────────────────────
        all_landmarks = tracker.find_hands(frame, draw=True)

        current_gesture = None
        hand_center     = None

        if all_landmarks:
            # Process the first detected hand for gesture classification
            landmarks = all_landmarks[0]

            # Detect gesture from normalized landmarks
            current_gesture = detect_gesture(landmarks)

            # Get pixel-space center for placing effects
            pixel_lms  = tracker.get_pixel_landmarks(landmarks, frame.shape)
            hand_center = tracker.get_hand_center(pixel_lms)

        # ── Gesture Smoothing ─────────────────────────────────────────
        if current_gesture:
            gesture_label = current_gesture
            gesture_hold  = HOLD_FRAMES
        else:
            if gesture_hold > 0:
                gesture_hold -= 1
            else:
                gesture_label = None

        # ── Visual Effects ────────────────────────────────────────────
        if gesture_label and hand_center:
            draw_gesture_effects(frame, gesture_label, hand_center, effect_state)
            draw_gesture_label(frame, gesture_label)
        else:
            # Reset animation when no gesture is active
            if gesture_hold == 0:
                effect_state.radius = 0
                effect_state.alpha  = 1.0
            draw_idle_hint(frame)

        # ── FPS Counter ───────────────────────────────────────────────
        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        draw_fps(frame, fps)

        # ── Title Bar ─────────────────────────────────────────────────
        cv2.putText(
            frame,
            "JJK Gesture System  |  ESC to quit",
            (frame.shape[1] - 340, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (180, 180, 180), 1
        )

        # ── Display ───────────────────────────────────────────────────
        cv2.imshow("JJK – Anime Hand Gesture Recognition", frame)

        # ── Key Handling ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27:          # ESC
            print("[INFO] Exiting.")
            break
        elif key == ord('r'):  # 'r' to reset effects
            effect_state = EffectState()
            print("[INFO] Effect state reset.")

    # ── Cleanup ──────────────────────────────────────────────────────
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("[INFO] Resources released. Goodbye!")


if __name__ == "__main__":
    main()