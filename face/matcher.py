"""
Face matching module.
Compares face encodings and determines if they match.
"""
import numpy as np


class FaceMatcher:
    """Match face encodings against stored database."""
    
    def __init__(self, threshold=0.6):
        """
        Initialize face matcher.
        
        Args:
            threshold (float): Maximum distance for a match (lower = stricter)
                             Typical values: 0.4-0.8 depending on accuracy needed
        """
        self.threshold = threshold
    
    def calculate_distance(self, encoding1, encoding2):
        """
        Calculate distance between two face encodings.
        Uses Euclidean distance.
        
        Args:
            encoding1 (numpy.ndarray): First face encoding
            encoding2 (numpy.ndarray): Second face encoding
            
        Returns:
            float: Distance between encodings (0 = identical, higher = more different)
        """
        if encoding1 is None or encoding2 is None:
            return float('inf')
        
        # Euclidean distance
        distance = np.linalg.norm(encoding1 - encoding2)
        return distance
    
    def calculate_cosine_similarity(self, encoding1, encoding2):
        """
        Calculate cosine similarity between two face encodings.
        
        Args:
            encoding1 (numpy.ndarray): First face encoding
            encoding2 (numpy.ndarray): Second face encoding
            
        Returns:
            float: Cosine similarity (1 = identical, 0 = orthogonal, -1 = opposite)
        """
        if encoding1 is None or encoding2 is None:
            return -1.0
        
        # Cosine similarity
        dot_product = np.dot(encoding1, encoding2)
        norm1 = np.linalg.norm(encoding1)
        norm2 = np.linalg.norm(encoding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return similarity
    
    def match_face(self, query_encoding, known_encodings, known_names):
        """
        Match a face encoding against database of known encodings.
        
        Args:
            query_encoding (numpy.ndarray): Encoding to match
            known_encodings (numpy.ndarray): Array of known encodings (N x encoding_dim)
            known_names (list): List of names corresponding to known_encodings
            
        Returns:
            dict: Match result containing:
                - matched (bool): Whether a match was found
                - name (str): Name of matched person (None if no match)
                - confidence (float): Confidence of match (0-1, higher = better)
                - distance (float): Distance to matched encoding
                - index (int): Index of matched encoding (None if no match)
        """
        if query_encoding is None or len(known_encodings) == 0:
            return {
                'matched': False,
                'name': None,
                'confidence': 0.0,
                'distance': float('inf'),
                'index': None
            }
        
        # Ensure known_encodings is 2D
        if len(known_encodings.shape) == 1:
            known_encodings = known_encodings.reshape(1, -1)
        
        # Validate names list matches encodings
        if len(known_names) != len(known_encodings):
            print(f"Warning: Mismatch between encodings ({len(known_encodings)}) and names ({len(known_names)})")
            return {
                'matched': False,
                'name': None,
                'confidence': 0.0,
                'distance': float('inf'),
                'index': None
            }
        
        # Calculate distances to all known faces
        distances = []
        for known_encoding in known_encodings:
            distance = self.calculate_distance(query_encoding, known_encoding)
            distances.append(distance)
        
        distances = np.array(distances)
        
        # Find best match
        best_match_idx = np.argmin(distances)
        best_distance = distances[best_match_idx]
        
        # Check if within threshold
        if best_distance <= self.threshold:
            # Calculate confidence (convert distance to 0-1 scale)
            # Lower distance = higher confidence
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
    
    def match_multiple(self, query_encodings, known_encodings, known_names):
        """
        Match multiple face encodings.
        
        Args:
            query_encodings (list): List of encodings to match
            known_encodings (numpy.ndarray): Array of known encodings
            known_names (list): List of names
            
        Returns:
            list: List of match results
        """
        results = []
        for query_encoding in query_encodings:
            result = self.match_face(query_encoding, known_encodings, known_names)
            results.append(result)
        return results
    
    def set_threshold(self, threshold):
        """
        Update matching threshold.
        
        Args:
            threshold (float): New threshold value
        """
        self.threshold = threshold
    
    def get_threshold(self):
        """Get current threshold."""
        return self.threshold
