# Face Authentication Attendance System

A real-time face recognition and authentication system for automated attendance tracking. Uses OpenCV for face detection and custom encoding algorithms for matching, with persistent storage for face data and attendance logs.

## Features
- ✅ **Real-time Face Detection** - OpenCV Haar Cascade for fast, accurate detection
- ✅ **Custom Face Encoding** - 128-dimensional feature vectors (histogram + spatial features)
- ✅ **Face Registration** - Interactive registration with unique ID assignment
- ✅ **Face Matching** - Euclidean distance-based matching with adjustable threshold
- ✅ **Persistent Storage** - JSON metadata + NumPy binary encodings
- 🚧 **Attendance Logging** - Automated punch-in/punch-out tracking (in progress)
- 🚧 **Anti-spoofing** - Liveness detection to prevent photo/video attacks (planned)

## Requirements
- Python 3.11+
- Webcam/Camera access

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/shivam1342/Face-Authentication-Attendance-System.git
cd Face-Authentication-Attendance-System
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
```

## Usage

### Keyboard Controls
- **`r`** - Register new face (enter name when prompted)
- **`l`** - List all registered faces
- **`t`** - Adjust matching threshold (default: 0.6)
- **`q`** - Quit application

### Registration Flow
1. Launch app: `python app.py`
2. Press `r` when your face is visible
3. Enter your name in the terminal
4. System saves your face encoding automatically

### Recognition Flow
1. Face appears in frame → System detects and matches
2. **GREEN box + "MATCH: [Name]"** - Face recognized
3. **RED box + "NO MATCH"** - Unknown face
4. Confidence and distance metrics displayed

## Project Structure
```
face-auth-attendance/
├── app.py                      # Main application entry point
├── camera/
│   └── camera.py              # Webcam abstraction with context manager
├── face/
│   ├── detector.py            # Face detection (Haar Cascade)
│   ├── encoder.py             # Custom 128-dim encoding
│   └── matcher.py             # Face matching with threshold
├── attendance/
│   ├── storage.py             # Face/attendance data persistence
│   └── attendance.py          # Attendance tracking logic
├── data/
│   ├── registered_faces/      # Face metadata + encodings
│   │   ├── faces.json         # Names, IDs, timestamps
│   │   └── encodings.npy      # 128-dim NumPy arrays
│   └── attendance_logs.json   # Daily attendance records
├── spoof/
│   └── liveness.py            # Anti-spoofing (planned)
└── utils/
    ├── config.py              # Configuration constants
    └── image_utils.py         # Image processing helpers
```

## Technical Details

### Face Detection
- Uses OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`)
- Returns bounding boxes: `(x, y, width, height)`
- No external dependencies or model downloads required

### Face Encoding
- Custom 128-dimensional feature vector:
  - **64 features**: Histogram of equalized grayscale image
  - **64 features**: 8×8 spatial grid features (LBP-like approach)
- Face resized to 128×128 for consistency
- Stored as NumPy arrays (`.npy` format)

### Face Matching
- **Metric**: Euclidean distance between encodings
- **Threshold**: Default 0.6 (adjustable via 't' key)
- **Confidence**: Calculated as `1 - (distance / threshold)`
- Returns: `matched`, `name`, `confidence`, `distance`, `index`

### Storage Format
**faces.json** (metadata):
```json
[
  {
    "name": "shivam",
    "id": 0,
    "timestamp": "2026-01-28T22:30:15.123456"
  }
]
```
**encodings.npy** (binary):
- Shape: `(n_faces, 128)`
- Data type: `float64`
- Indexed to match faces.json order

## Troubleshooting

### Face not detected?
- Ensure good lighting
- Face camera directly
- Move closer to camera
- Check if Haar Cascade model loads correctly

### Matching threshold too strict?
- Press `t` to adjust threshold (try 0.7-1.0)
- Current threshold displayed in terminal

### Data corruption?
- Clear storage: `Remove-Item data\registered_faces\* -Force` (Windows)
- Or: `rm -rf data/registered_faces/*` (Linux/Mac)
- Re-register all faces

## Roadmap
- [ ] Attendance logging integration
- [ ] Duplicate attendance prevention
- [ ] Liveness detection (blink/head movement)
- [ ] Multi-face registration in single frame
- [ ] Web dashboard for attendance reports
- [ ] Export attendance to CSV/Excel

## Contributing
Pull requests welcome! Please ensure code follows existing structure and includes error handling.

## License
MIT License - see LICENSE file for details

## Author
Shivam - [GitHub](https://github.com/shivam1342)
