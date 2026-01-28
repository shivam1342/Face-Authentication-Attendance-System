"""
Face detection module using OpenCV DNN.
Detects faces in frames and returns bounding boxes.
"""
import cv2
import numpy as np
import os


class FaceDetector:
    """Face detector using OpenCV DNN with Caffe model."""
    
    def __init__(self, min_detection_confidence=0.7):
        """
        Initialize face detector.
        
        Args:
            min_detection_confidence (float): Minimum confidence threshold (0.0 to 1.0)
        """
        self.min_confidence = min_detection_confidence
        
        # Load OpenCV's DNN face detector
        # Using pre-trained Caffe model (built into OpenCV)
        model_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(model_file)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load face detection model")
    
    def detect(self, frame):
        """
        Detect faces in frame.
        
        Args:
            frame (numpy.ndarray): BGR image frame from camera
            
        Returns:
            list: List of face detections, each containing:
                - bbox: (x, y, w, h) bounding box
                - confidence: detection confidence score
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        detected_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces = []
        
        for (x, y, w, h) in detected_faces:
            # Haar cascades don't provide confidence, so we use 1.0
            faces.append({
                'bbox': (x, y, w, h),
                'confidence': 1.0  # Haar cascade doesn't provide confidence
            })
        
        return faces
    
    def draw_detections(self, frame, faces):
        """
        Draw bounding boxes and info on frame.
        
        Args:
            frame (numpy.ndarray): Image frame
            faces (list): List of face detections from detect()
            
        Returns:
            numpy.ndarray: Frame with drawings
        """
        for face in faces:
            x, y, w, h = face['bbox']
            confidence = face['confidence']
            
            # Draw rectangle
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence
            text = f"{confidence:.2f}"
            cv2.putText(frame, text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def get_face_region(self, frame, bbox):
        """
        Extract face region from frame.
        
        Args:
            frame (numpy.ndarray): Full image frame
            bbox (tuple): Bounding box (x, y, w, h)
            
        Returns:
            numpy.ndarray: Cropped face region
        """
        x, y, w, h = bbox
        face_region = frame[y:y+h, x:x+w]
        return face_region
    
    def close(self):
        """Release resources."""
        pass  # Haar cascade doesn't need explicit cleanup
