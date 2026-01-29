"""
Basic heuristic-based liveness detection using blink detection.

NOT a security-grade anti-spoofing system.
Designed for controlled attendance scenarios.
Can be defeated by replay attacks or high-quality spoofs.
"""

import cv2
import time


class LivenessDetector:
    """
    Heuristic-based liveness detection using eye visibility changes.
    Uses Haar eye detection instead of landmark-based EAR.
    """

    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        # Timing-based blink detection
        self.min_closed_duration = 0.15  # seconds
        self.max_check_duration = 1.5    # seconds

        self.reset()

    def reset(self):
        self.last_eyes_visible = True
        self.eye_closed_start = None
        self.check_start_time = None
        self.blink_detected = False

    def detect_eyes(self, face_region):
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        return self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )

    def check_blink(self, frame, face_bbox):
        x, y, w, h = face_bbox
        face_region = frame[y:y + h, x:x + w]

        eyes = self.detect_eyes(face_region)
        eyes_visible = len(eyes) >= 2
        current_time = time.time()

        # Eyes just closed
        if not eyes_visible and self.last_eyes_visible:
            self.eye_closed_start = current_time

        # Eyes just opened → blink
        if eyes_visible and not self.last_eyes_visible:
            if self.eye_closed_start:
                closed_duration = current_time - self.eye_closed_start
                if closed_duration >= self.min_closed_duration:
                    self.blink_detected = True

            self.eye_closed_start = None

        self.last_eyes_visible = eyes_visible

        return {
            "eyes_detected": eyes_visible,
            "blink_detected": self.blink_detected
        }

    def verify_liveness(self, frame, face_bbox):
        if self.check_start_time is None:
            self.check_start_time = time.time()

        elapsed = time.time() - self.check_start_time

        if elapsed > self.max_check_duration:
            return {
                "is_live": False,
                "time_elapsed": elapsed,
                "message": "Liveness check timed out"
            }

        result = self.check_blink(frame, face_bbox)

        if self.blink_detected:
            return {
                "is_live": True,
                "time_elapsed": elapsed,
                "message": "Liveness verified (blink detected)"
            }

        if not result["eyes_detected"]:
            return {
                "is_live": None,
                "time_elapsed": elapsed,
                "message": "Eyes not visible, please face the camera"
            }

        return {
            "is_live": None,
            "time_elapsed": elapsed,
            "message": "Please blink naturally"
        }
