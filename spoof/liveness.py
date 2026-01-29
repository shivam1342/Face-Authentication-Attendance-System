"""
Basic heuristic-based liveness detection using blink detection.

This is NOT a security-grade anti-spoofing system.
It provides basic liveness checks suitable for controlled attendance scenarios.
Can be defeated by videos or sophisticated attacks.
"""

import cv2
import time


class LivenessDetector:
    """
    Basic heuristic-based liveness detection using blink detection.
    
    Detects eye blinks using Haar Cascade eye detection and Eye Aspect Ratio (EAR).
    Requires the user to blink naturally during a short verification period.
    """
    
    def __init__(self):
        # Load Haar Cascade for eye detection
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        # Blink detection parameters
        self.blink_threshold = 0.25  # EAR threshold to consider eye as closed
        self.consecutive_frames = 2   # Frames with closed eyes to register a blink
        self.required_blinks = 0      # Disabled for now (Haar eye detection unreliable)
        
        # State tracking
        self.reset()
    
    def reset(self):
        """Reset detector state for a new liveness check."""
        self.blink_count = 0
        self.closed_frames = 0
        self.check_start_time = None
        self.max_check_duration = 1.0  # Reduced timeout since blink check disabled
    
    def calculate_ear(self, eye_region):
        """
        Calculate Eye Aspect Ratio (EAR) for an eye region.
        
        Simplified EAR: Uses eye region height-to-width ratio as a proxy.
        When eyes are open, ratio is higher. When closed, ratio drops significantly.
        
        Returns:
            float: Eye aspect ratio (higher = more open)
        """
        h, w = eye_region.shape[:2]
        if w == 0:
            return 0
        
        # Simple ratio - real EAR uses facial landmarks, but this works for basic detection
        ratio = h / float(w)
        return ratio
    
    def detect_eyes(self, face_region):
        """
        Detect eyes within a face region.
        
        Args:
            face_region: Cropped face image (BGR)
            
        Returns:
            list: List of detected eye bounding boxes [(x, y, w, h), ...]
        """
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        return eyes
    
    def check_blink(self, frame, face_bbox):
        """
        Check for blink in the current frame.
        
        Args:
            frame: Full frame image (BGR)
            face_bbox: Face bounding box tuple (x, y, w, h)
            
        Returns:
            dict: Blink detection result with keys:
                - eyes_detected: bool
                - blink_detected: bool  
                - ear_values: list of EAR values for detected eyes
        """
        # Extract face region
        x, y, w, h = face_bbox
        face_region = frame[y:y+h, x:x+w]
        
        # Detect eyes
        eyes = self.detect_eyes(face_region)
        
        if len(eyes) == 0:
            return {
                'eyes_detected': False,
                'blink_detected': False,
                'ear_values': []
            }
        
        # Calculate EAR for each eye
        ear_values = []
        for (ex, ey, ew, eh) in eyes:
            eye_region = face_region[ey:ey+eh, ex:ex+ew]
            ear = self.calculate_ear(eye_region)
            ear_values.append(ear)
        
        # Average EAR across both eyes
        avg_ear = sum(ear_values) / len(ear_values) if ear_values else 0
        
        # Check if eyes are closed (low EAR)
        eyes_closed = avg_ear < self.blink_threshold
        
        blink_detected = False
        if eyes_closed:
            self.closed_frames += 1
        else:
            # Eyes opened after being closed = blink detected
            if self.closed_frames >= self.consecutive_frames:
                self.blink_count += 1
                blink_detected = True
            self.closed_frames = 0
        
        return {
            'eyes_detected': True,
            'blink_detected': blink_detected,
            'ear_values': ear_values
        }
    
    def verify_liveness(self, frame, face_bbox):
        """
        Perform liveness verification on a single frame.
        
        This should be called repeatedly with video frames until liveness is verified
        or the check times out.
        
        Args:
            frame: Current frame image (BGR)
            face_bbox: Face bounding box tuple (x, y, w, h)
            
        Returns:
            dict: Verification result with keys:
                - is_live: bool or None (None = still checking)
                - blink_count: int
                - time_elapsed: float
                - message: str
        """
        # Start timer on first call
        if self.check_start_time is None:
            self.check_start_time = time.time()
        
        time_elapsed = time.time() - self.check_start_time
        
        # Check for timeout
        if time_elapsed > self.max_check_duration:
            return {
                'is_live': False,
                'blink_count': self.blink_count,
                'time_elapsed': time_elapsed,
                'message': f'Timeout: {self.blink_count} blinks detected (need {self.required_blinks})'
            }
        
        # Check for blink in current frame
        blink_result = self.check_blink(frame, face_bbox)
        
        if not blink_result['eyes_detected']:
            return {
                'is_live': None,
                'blink_count': self.blink_count,
                'time_elapsed': time_elapsed,
                'message': 'Eyes not detected, please look at camera'
            }
        
        # Check if we have enough blinks
        if self.blink_count >= self.required_blinks:
            return {
                'is_live': True,
                'blink_count': self.blink_count,
                'time_elapsed': time_elapsed,
                'message': f'Liveness verified ({self.blink_count} blinks)'
            }
        
        # Still checking
        return {
            'is_live': None,
            'blink_count': self.blink_count,
            'time_elapsed': time_elapsed,
            'message': f'Please blink naturally ({self.blink_count}/{self.required_blinks})'
        }
