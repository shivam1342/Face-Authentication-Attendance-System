"""
Attendance management module.
Implements punch-in and punch-out logic based on successful authentication.
Ensures attendance events are recorded with proper timestamps and user identity.
"""


class AttendanceManager:
    """
    Manage attendance operations (punch-in/punch-out).
    """
    
    def __init__(self, logger):
        """
        Initialize attendance manager.
        
        Args:
            logger: AttendanceLogger instance
        """
        self.logger = logger
    
    def punch_in(self, name, face_id):
        """
        Record punch-in for authenticated user.
        
        Args:
            name: User's name
            face_id: User's face ID
            
        Returns:
            dict: Result with success and message
        """
        return self.logger.punch_in(name, face_id)
    
    def punch_out(self, name, face_id):
        """
        Record punch-out for authenticated user.
        
        Args:
            name: User's name
            face_id: User's face ID
            
        Returns:
            dict: Result with success and message
        """
        return self.logger.punch_out(name, face_id)
    
    def get_status_today(self, name):
        """
        Get current status for a person today.
        
        Args:
            name: Person's name
            
        Returns:
            str: Status ('in', 'out', or None)
        """
        return self.logger.get_status_today(name)
    
    def get_today_summary(self):
        """
        Get summary of all attendance today.
        
        Returns:
            list: Summary of all people's attendance
        """
        return self.logger.get_today_summary()

