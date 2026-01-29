"""
Handles webcam initialization and frame capture.
Abstracts camera access so the rest of the system remains independent of the video source.
"""
import cv2


class Camera:
    """Webcam abstraction for frame capture."""
    
    def __init__(self, camera_index=1):
        """
        Initialize camera.
        
        Args:
            camera_index: Camera device index (0 for default webcam)
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
        Capture a frame from camera.
        
        Returns:
            tuple: (success, frame)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        success, frame = self.cap.read()
        return success, frame
    
    def release(self):
        """Release camera resource."""
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
