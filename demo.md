# Demo Guide

## How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Register Faces**
   - Add face embeddings to `data/registered_faces/`

3. **Start the System**
   ```bash
   python app.py
   ```

4. **Test Attendance**
   - Stand in front of camera
   - System will detect face and mark attendance
   - Check logs in `data/attendance_logs.json`

## Demo Steps

1. Face detection verification
2. Liveness check (blink detection)
3. Face matching against database
4. Attendance recording
