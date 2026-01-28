"""
Camera abstraction module for webcam capture.
Handles opening, capturing, and releasing the camera.
"""
import cv2


class Camera:
    """Simple webcam abstraction for frame capture."""
    
    def __init__(self, camera_index=0):
        """
        Initialize camera.
        
        Args:
            camera_index (int): Camera device index (default: 0 for primary webcam)
        """
        self.camera_index = camera_index
        self.cap = None
        
    def open(self):
        """Open the camera."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        return True
    
    def read_frame(self):
        """
        Capture a single frame from the camera.
        
        Returns:
            tuple: (success, frame) where success is bool and frame is numpy array
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        success, frame = self.cap.read()
        return success, frame
    
    def is_opened(self):
        """Check if camera is opened."""
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
