"""
Extracts a compact face representation from detected face regions.
Used during both registration and identification to ensure consistent face matching.
"""
import cv2
import numpy as np


class FaceEncoder:
    """Face encoder using histogram and spatial features."""
    
    def __init__(self):
        """Initialize face encoder."""
        pass
    
    def encode(self, frame, bbox):
        """
        Generate face embedding from frame.
        
        Args:
            frame: BGR image frame
            bbox: Bounding box (x, y, w, h) of face region
            
        Returns:
            numpy.ndarray: 128-dimensional face encoding vector
        """
        x, y, w, h = bbox
        
        # Ensure coordinates are within bounds
        x, y = max(0, x), max(0, y)
        h_max, w_max = frame.shape[:2]
        face_region = frame[y:min(y+h, h_max), x:min(x+w, w_max)]
        
        if face_region.size == 0:
            return None
        
        # Resize to standard size
        face_resized = cv2.resize(face_region, (128, 128))
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Histogram equalization
        gray = cv2.equalizeHist(gray)
        
        # Extract features
        features = []
        
        # Histogram features (64 bins)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)
        
        # Spatial features (8x8 grid = 64 values)
        for i in range(8):
            for j in range(8):
                block = gray[i*16:(i+1)*16, j*16:(j+1)*16]
                features.append(block.mean() / 255.0)
        
        encoding = np.array(features, dtype=np.float32)
        return encoding



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
