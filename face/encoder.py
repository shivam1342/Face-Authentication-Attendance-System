"""
Face encoding module using face landmarks.
Converts face images to embedding vectors for recognition.
"""
import cv2
import numpy as np


class FaceEncoder:
    """Face encoder using OpenCV and geometric features."""
    
    def __init__(self):
        """Initialize face encoder."""
        # Load face landmark detector (68 points)
        landmark_model = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_cascade = cv2.CascadeClassifier(landmark_model)
    
    def encode(self, frame, bbox=None):
        """
        Generate face embedding from frame.
        
        Args:
            frame (numpy.ndarray): BGR image frame
            bbox (tuple): Optional bounding box (x, y, w, h) to crop face region
            
        Returns:
            numpy.ndarray: Face embedding vector (simplified 128-dimension)
                          or None if encoding fails
        """
        # If bbox provided, crop to face region
        if bbox is not None:
            x, y, w, h = bbox
            # Ensure coordinates are within bounds
            x, y = max(0, x), max(0, y)
            h_max, w_max = frame.shape[:2]
            face_region = frame[y:min(y+h, h_max), x:min(x+w, w_max)]
        else:
            face_region = frame
        
        if face_region.size == 0:
            return None
        
        # Resize face to standard size for consistent encoding
        face_resized = cv2.resize(face_region, (128, 128))
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better consistency
        gray = cv2.equalizeHist(gray)
        
        # Extract features using multiple methods
        features = []
        
        # 1. Histogram features (64 values)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        features.extend(hist)
        
        # 2. LBP-like features (64 values)
        # Divide face into 8x8 grid and compute mean intensity
        for i in range(8):
            for j in range(8):
                block = gray[i*16:(i+1)*16, j*16:(j+1)*16]
                features.append(block.mean() / 255.0)
        
        # Convert to numpy array
        encoding = np.array(features, dtype=np.float32)
        
        return encoding
    
    def encode_multiple(self, frame, bboxes):
        """
        Encode multiple faces from frame.
        
        Args:
            frame (numpy.ndarray): BGR image frame
            bboxes (list): List of bounding boxes [(x, y, w, h), ...]
            
        Returns:
            list: List of encodings (one per face)
        """
        encodings = []
        for bbox in bboxes:
            encoding = self.encode(frame, bbox)
            encodings.append(encoding)
        return encodings
    
    def get_encoding_shape(self):
        """
        Get the shape of encoding vector.
        
        Returns:
            int: Dimension of encoding (128 dimensions)
        """
        return 128  # 64 histogram + 64 spatial features
    
    def close(self):
        """Release resources."""
        pass  # No resources to release


def calculate_similarity(encoding1, encoding2):
    """
    Calculate similarity between two face encodings.
    Uses Euclidean distance - lower is more similar.
    
    Args:
        encoding1 (numpy.ndarray): First face encoding
        encoding2 (numpy.ndarray): Second face encoding
        
    Returns:
        float: Distance between encodings (0 = identical)
    """
    if encoding1 is None or encoding2 is None:
        return float('inf')
    
    distance = np.linalg.norm(encoding1 - encoding2)
    return distance


def match_face(query_encoding, known_encodings, threshold=0.5):
    """
    Match query encoding against known encodings.
    
    Args:
        query_encoding (numpy.ndarray): Encoding to match
        known_encodings (list): List of known encodings
        threshold (float): Maximum distance to consider a match
        
    Returns:
        tuple: (match_index, distance) or (None, None) if no match
    """
    if query_encoding is None:
        return None, None
    
    best_match_idx = None
    best_distance = float('inf')
    
    for idx, known_encoding in enumerate(known_encodings):
        distance = calculate_similarity(query_encoding, known_encoding)
        
        if distance < best_distance and distance < threshold:
            best_distance = distance
            best_match_idx = idx
    
    return best_match_idx, best_distance
