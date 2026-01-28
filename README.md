# Face Authentication Attendance System

This is a face recognition-based attendance system with liveness detection.

## Features
- Face detection and recognition
- Anti-spoofing with liveness checks
- Automated punch-in/punch-out
- Attendance logging

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `python app.py`

## Project Structure
- `camera/` - Webcam capture
- `face/` - Detection, encoding, and matching
- `attendance/` - Attendance logic and storage
- `spoof/` - Liveness detection
- `utils/` - Helper utilities
- `data/` - Stored faces and logs
