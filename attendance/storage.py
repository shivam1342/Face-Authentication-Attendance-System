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
    
    def punch_in(self, name, face_id):
        """
        Log punch-in (arrival).
        
        Args:
            name (str): Person's name
            face_id (int): Face ID
            
        Returns:
            dict: Result with success status and message
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Check if already punched in within last 10 seconds
        status = self.get_status_recent(name)
        if status == 'in':
            return {
                'success': False,
                'message': f'{name} already punched in (wait 10s)',
                'type': 'duplicate'
            }
        elif status == 'out':
            return {
                'success': False,
                'message': f'{name} already completed attendance (wait 10s)',
                'type': 'already_completed'
            }
        
        entry = {
            'name': name,
            'face_id': int(face_id),  # Convert numpy int64 to Python int
            'type': 'punch_in',
            'timestamp': datetime.now().isoformat(),
            'date': today,
            'time': datetime.now().strftime('%H:%M:%S')
        }
        
        self.logs.append(entry)
        self._save_logs()
        
        return {
            'success': True,
            'message': f'✓ {name} punched in at {entry["time"]}',
            'entry': entry
        }
    
    def punch_out(self, name, face_id):
        """
        Log punch-out (departure).
        
        Args:
            name (str): Person's name
            face_id (int): Face ID
            
        Returns:
            dict: Result with success status and message
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Check current status (can punch out anytime after punch-in)
        status = self.get_status_today(name)
        if status is None:
            return {
                'success': False,
                'message': f'{name} has not punched in yet',
                'type': 'not_punched_in'
            }
        elif status == 'out':
            # Check if punch-out was recent (within 10 seconds)
            recent_status = self.get_status_recent(name)
            if recent_status == 'out':
                return {
                    'success': False,
                    'message': f'{name} already punched out (wait 10s)',
                    'type': 'duplicate'
                }
        
        # Get punch-in time for duration calculation
        punch_in_entry = self._get_last_punch_in_today(name)
        
        entry = {
            'name': name,
            'face_id': int(face_id),  # Convert numpy int64 to Python int
            'type': 'punch_out',
            'timestamp': datetime.now().isoformat(),
            'date': today,
            'time': datetime.now().strftime('%H:%M:%S')
        }
        
        # Calculate duration
        if punch_in_entry:
            in_time = datetime.fromisoformat(punch_in_entry['timestamp'])
            out_time = datetime.now()
            duration = out_time - in_time
            
            hours = duration.total_seconds() / 3600
            entry['duration_hours'] = round(hours, 2)
            entry['duration_str'] = str(duration).split('.')[0]  # HH:MM:SS format
        
        self.logs.append(entry)
        self._save_logs()
        
        message = f'✓ {name} punched out at {entry["time"]}'
        if 'duration_str' in entry:
            message += f' (Duration: {entry["duration_str"]})'
        
        return {
            'success': True,
            'message': message,
            'entry': entry
        }
    
    def get_status_recent(self, name):
        """
        Get current punch status for a person within last 10 seconds.
        
        Args:
            name (str): Person's name
            
        Returns:
            str or None: 'in' if punched in, 'out' if punched out, None if no activity
        """
        now = datetime.now()
        
        # Find the last entry for this person within last 10 seconds
        last_entry = None
        for log in reversed(self.logs):
            if log.get('name') == name:
                log_time = datetime.fromisoformat(log.get('timestamp'))
                time_diff = (now - log_time).total_seconds()
                if time_diff <= 10:
                    last_entry = log
                    break
        
        if last_entry is None:
            return None
        
        if last_entry.get('type') == 'punch_in':
            return 'in'
        elif last_entry.get('type') == 'punch_out':
            return 'out'
        
        return None
    
    def get_status_today(self, name):
        """
        Get current punch status for a person today.
        
        Args:
            name (str): Person's name
            
        Returns:
            str or None: 'in' if punched in, 'out' if punched out, None if no activity
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Find the last entry for this person today
        last_entry = None
        for log in reversed(self.logs):
            if log.get('name') == name and log.get('date') == today:
                last_entry = log
                break
        
        if last_entry is None:
            return None
        
        if last_entry.get('type') == 'punch_in':
            return 'in'
        elif last_entry.get('type') == 'punch_out':
            return 'out'
        
        return None
    
    def _get_last_punch_in_recent(self, name):
        """Get the last punch-in entry for a person within last 10 seconds."""
        now = datetime.now()
        
        for log in reversed(self.logs):
            if log.get('name') == name and log.get('type') == 'punch_in':
                log_time = datetime.fromisoformat(log.get('timestamp'))
                time_diff = (now - log_time).total_seconds()
                if time_diff <= 10:
                    return log
        
        return None
    
    def _get_last_punch_in_today(self, name):
        """Get the last punch-in entry for a person today."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        for log in reversed(self.logs):
            if (log.get('name') == name and 
                log.get('date') == today and 
                log.get('type') == 'punch_in'):
                return log
        
        return None
    
    def get_today_logs(self):
        """Get today's attendance logs."""
        today = datetime.now().strftime('%Y-%m-%d')
        return [log for log in self.logs if log.get('date') == today]
    
    def get_today_summary(self):
        """
        Get summary of today's attendance with punch-in/out status.
        
        Returns:
            list: List of dicts with name, status, punch_in_time, punch_out_time, duration
        """
        today = datetime.now().strftime('%Y-%m-%d')
        today_logs = self.get_today_logs()
        
        # Group by person
        people = {}
        for log in today_logs:
            name = log['name']
            if name not in people:
                people[name] = {
                    'name': name,
                    'punch_in': None,
                    'punch_out': None,
                    'status': None
                }
            
            if log.get('type') == 'punch_in':
                people[name]['punch_in'] = log['time']
            elif log.get('type') == 'punch_out':
                people[name]['punch_out'] = log['time']
                if 'duration_str' in log:
                    people[name]['duration'] = log['duration_str']
        
        # Set status
        for name, data in people.items():
            if data['punch_in'] and data['punch_out']:
                data['status'] = 'Completed'
            elif data['punch_in']:
                data['status'] = 'Checked In'
            else:
                data['status'] = 'Unknown'
        
        return list(people.values())
