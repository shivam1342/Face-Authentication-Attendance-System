"""
Attendance management module.
Handles punch-in/punch-out logic and attendance status tracking.
"""

from datetime import datetime


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
        Process punch-in for a person.
        
        Args:
            name (str): Person's name
            face_id (int): Face ID
            
        Returns:
            dict: Result with success, message, and entry
        """
        return self.logger.punch_in(name, face_id)
    
    def punch_out(self, name, face_id):
        """
        Process punch-out for a person.
        
        Args:
            name (str): Person's name
            face_id (int): Face ID
            
        Returns:
            dict: Result with success, message, and entry
        """
        return self.logger.punch_out(name, face_id)
    
    def get_status(self, name):
        """
        Get current status for a person today.
        
        Args:
            name (str): Person's name
            
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
    
    def display_summary(self):
        """
        Display today's attendance summary in a readable format.
        """
        summary = self.get_today_summary()
        
        if not summary:
            print("\n📋 No attendance records today")
            return
        
        print("\n" + "="*70)
        print("📋 TODAY'S ATTENDANCE SUMMARY")
        print("="*70)
        print(f"{'Name':<20} {'Status':<15} {'Punch In':<12} {'Punch Out':<12} {'Duration':<10}")
        print("-"*70)
        
        for record in summary:
            name = record['name'] or '-'
            status = record['status'] or '-'
            punch_in = record.get('punch_in') or '-'
            punch_out = record.get('punch_out') or '-'
            duration = record.get('duration') or '-'
            
            print(f"{name:<20} {status:<15} {punch_in:<12} {punch_out:<12} {duration:<10}")
        
        print("="*70)

