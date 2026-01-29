"""
Detects faces in video frames using OpenCV Haar cascades.
Returns bounding boxes for downstream face processing and identification.
"""
import cv2


class FaceDetector:
    """Face detector using OpenCV Haar cascades."""
    
    def __init__(self, min_detection_confidence=0.7):
        """
        Initialize face detector.
        
        Args:
            min_detection_confidence: Minimum confidence threshold
        """
        self.min_confidence = min_detection_confidence
        
        # Load Haar cascade
        model_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(model_file)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade")
    
    def detect(self, frame):
        """
        Detect faces in frame.
        
        Args:
            frame: BGR image frame
            
        Returns:
            list: List of detections with bbox (x, y, w, h) and confidence
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detected_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        faces = []
        for (x, y, w, h) in detected_faces:
            faces.append({
                'bbox': (x, y, w, h),
                'confidence': 1.0
            })
        
        return faces
    
    def draw_detections(self, frame, faces):
        """
        Draw bounding boxes on frame.
        
        Args:
            frame: Image frame
            faces: List of face detections
            
        Returns:
            Frame with bounding boxes drawn
        """
        for face in faces:
            x, y, w, h = face['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return frame

