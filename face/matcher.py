"""
Compares live face representations against registered faces using similarity thresholds.
Determines whether the current face matches a known identity.
"""
import numpy as np


class FaceMatcher:
    """Match face encodings against registered database."""
    
    def __init__(self, threshold=8.0):
        """
        Initialize face matcher.
        
        Args:
            threshold: Maximum distance for a match (lower = stricter)
        """
        self.threshold = threshold
    
    def match_face(self, query_encoding, known_encodings, known_names):
        """
        Match face encoding against known encodings.
        
        Args:
            query_encoding: Encoding to match
            known_encodings: Array of known encodings
            known_names: List of names corresponding to encodings
            
        Returns:
            dict: Match result with matched, name, confidence, distance, index
        """
        if query_encoding is None or len(known_encodings) == 0:
            return {
                'matched': False,
                'name': None,
                'confidence': 0.0,
                'distance': float('inf'),
                'index': None
            }
        
        # Ensure 2D array
        if len(known_encodings.shape) == 1:
            known_encodings = known_encodings.reshape(1, -1)
        
        # Calculate Euclidean distances
        distances = []
        for known_encoding in known_encodings:
            distance = np.linalg.norm(query_encoding - known_encoding)
            distances.append(distance)
        
        distances = np.array(distances)
        
        # Find best match
        best_match_idx = np.argmin(distances)
        best_distance = distances[best_match_idx]
        
        # Check threshold
        if best_distance <= self.threshold:
            confidence = max(0.0, 1.0 - (best_distance / self.threshold))
            
            return {
                'matched': True,
                'name': known_names[best_match_idx],
                'confidence': confidence,
                'distance': best_distance,
                'index': best_match_idx
            }
        else:
            return {
                'matched': False,
                'name': None,
                'confidence': 0.0,
                'distance': best_distance,
                'index': None
            }

