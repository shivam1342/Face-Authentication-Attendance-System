"""
Storage module for face encodings and attendance logs.
Handles saving/loading registered faces and attendance records.
"""
import json
import numpy as np
import os
from datetime import datetime


class FaceStorage:
    """Manage registered face encodings."""
    
    def __init__(self, storage_dir="data/registered_faces"):
        """
        Initialize face storage.
        
        Args:
            storage_dir (str): Directory to store face data
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self.faces_file = os.path.join(storage_dir, "faces.json")
        self.encodings_file = os.path.join(storage_dir, "encodings.npy")
        
        # Load existing data
        self.faces = self._load_faces()
        self.encodings = self._load_encodings()
    
    def _load_faces(self):
        """Load face metadata (names, IDs, etc.)."""
        if os.path.exists(self.faces_file):
            try:
                with open(self.faces_file, 'r') as f:
                    content = f.read().strip()
                    if not content:  # Empty file
                        return []
                    return json.loads(content)
            except (json.JSONDecodeError, ValueError):
                # Corrupted file, return empty list
                return []
        return []
    
    def _load_encodings(self):
        """Load face encodings."""
        if os.path.exists(self.encodings_file):
            try:
                return np.load(self.encodings_file)
            except (ValueError, IOError):
                # Corrupted file, return empty array
                return np.array([])
        return np.array([])
    
    def _save_faces(self):
        """Save face metadata."""
        with open(self.faces_file, 'w') as f:
            json.dump(self.faces, f, indent=2)
    
    def _save_encodings(self):
        """Save face encodings."""
        np.save(self.encodings_file, self.encodings)
    
    def register_face(self, name, encoding):
        """
        Register a new face.
        
        Args:
            name (str): Person's name
            encoding (numpy.ndarray): Face encoding vector
            
        Returns:
            int: ID of registered face
        """
        # Generate ID
        face_id = len(self.faces)
        
        # Add metadata
        face_data = {
            'id': face_id,
            'name': name,
            'registered_at': datetime.now().isoformat()
        }
        self.faces.append(face_data)
        
        # Add encoding
        if len(self.encodings) == 0:
            self.encodings = encoding.reshape(1, -1)
        else:
            self.encodings = np.vstack([self.encodings, encoding])
        
        # Save to disk
        self._save_faces()
        self._save_encodings()
        
        return face_id
    
    def get_all_encodings(self):
        """
        Get all registered encodings.
        
        Returns:
            numpy.ndarray: Array of encodings
        """
        return self.encodings
    
    def get_all_names(self):
        """
        Get all registered names.
        
        Returns:
            list: List of names
        """
        return [face['name'] for face in self.faces]
    
    def get_face_by_index(self, index):
        """
        Get face data by index.
        
        Args:
            index (int): Face index
            
        Returns:
            dict: Face data or None
        """
        if 0 <= index < len(self.faces):
            return self.faces[index]
        return None
    
    def count(self):
        """Get number of registered faces."""
        return len(self.faces)
    
    def list_all(self):
        """
        List all registered faces.
        
        Returns:
            list: List of face metadata
        """
        return self.faces.copy()


class AttendanceLogger:
    """Log attendance records."""
    
    def __init__(self, log_file="data/attendance_logs.json"):
        """
        Initialize attendance logger.
        
        Args:
            log_file (str): Path to attendance log file
        """
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        self.logs = self._load_logs()
    
    def _load_logs(self):
        """Load existing logs."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    content = f.read().strip()
                    if not content:  # Empty file
                        return []
                    return json.loads(content)
            except (json.JSONDecodeError, ValueError):
                # Corrupted file, return empty list
                return []
        return []
    
    def _save_logs(self):
        """Save logs to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def log_attendance(self, name, face_id):
        """
        Log attendance entry.
        
        Args:
            name (str): Person's name
            face_id (int): Face ID
            
        Returns:
            dict: Log entry
        """
        entry = {
            'name': name,
            'face_id': face_id,
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S')
        }
        
        self.logs.append(entry)
        self._save_logs()
        
        return entry
    
    def get_today_logs(self):
        """Get today's attendance logs."""
        today = datetime.now().strftime('%Y-%m-%d')
        return [log for log in self.logs if log.get('date') == today]
    
    def has_logged_today(self, name):
        """
        Check if person has already logged attendance today.
        
        Args:
            name (str): Person's name
            
        Returns:
            bool: True if already logged
        """
        today = datetime.now().strftime('%Y-%m-%d')
        for log in self.logs:
            if log.get('name') == name and log.get('date') == today:
                return True
        return False
