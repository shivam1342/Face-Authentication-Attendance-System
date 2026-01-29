# Face Authentication Attendance System

A real-time face recognition and authentication system for automated attendance tracking. Uses OpenCV Haar Cascades for face detection with custom encoding algorithms for matching, and lightweight JSON-based persistent storage.

## System Architecture

### Core Components
- **Face Detection**: OpenCV Haar Cascade classifier
- **Face Encoding**: Custom heuristic-based 128-dimensional feature vectors (histogram + spatial features)
- **Face Matching**: Euclidean distance with configurable threshold
- **Liveness Detection**: Basic blink-based anti-spoofing (heuristic)
- **Storage**: JSON for metadata, NumPy binary for encodings
- **Attendance**: Timestamp-based punch-in/out with 10-second cooldown

### Model Choices & Rationale

**Why Haar Cascades over Deep Learning?**
- ✅ Zero external dependencies (built into OpenCV)
- ✅ Fast inference on CPU (~30-60 FPS)
- ✅ No model downloads or training required
- ✅ Sufficient accuracy for controlled indoor environments
- ❌ Less robust than CNNs in varying lighting/angles
- ❌ Higher false positive rate outdoors

**Why Custom Encoding over Pre-trained Embeddings?**
- ✅ Lightweight (128 dimensions vs 512+ for FaceNet/ArcFace)
- ✅ Transparent feature extraction (histogram + grid)
- ✅ No GPU or large model files required
- ❌ Lower accuracy than deep embeddings (~85% vs 99%+)
- ❌ Sensitive to lighting variations

**Why Blink Detection over Advanced Liveness?**
- ✅ Simple implementation with eye cascade
- ✅ Works without IR cameras or depth sensors
- ✅ Sufficient for trusted environment scenarios
- ❌ **EASILY DEFEATED by videos or eye cutouts**
- ❌ Not production-grade security

## Design Decisions

### 10-Second Cooldown Period
- Prevents accidental duplicate registrations
- Allows multiple punch-in/out cycles for testing
- **Production should use full-day restrictions**

### One Punch-in/Out Cycle
- Current: Can repeat every 10 seconds
- **Production constraint**: Should enforce one cycle per work day
- Stored in `attendance/storage.py` - modify `get_status_recent()` to `get_status_today()`

### JSON Storage
- Human-readable for debugging and transparency
- Easy data inspection without tools
- **Not suitable for 1000+ users** (use PostgreSQL/MongoDB for scale)

## Known Limitations

### Face Detection
- ❌ Fails with face masks, sunglasses, or hats
- ❌ Poor performance in low light (<100 lux)
- ❌ Struggles with side profiles (>30° angle)
- ❌ Cannot detect faces smaller than 30×30 pixels

### Face Recognition
- ❌ Accuracy drops with facial hair changes
- ❌ Identical twins may match incorrectly
- ❌ Aging affects matching (re-register yearly)
- ❌ Threshold tuning required per environment

### Liveness Detection
- ❌ **NOT SECURE** - can be bypassed with:
  - Photo of person blinking (video)
  - Printed face with eye cutouts
  - Screen replay attacks
- ❌ No depth sensing or challenge-response
- ❌ Eye cascade unreliable with glasses/makeup

### Storage & Performance
- ❌ Linear search O(n) for face matching
- ❌ JSON parsing overhead for large datasets
- ❌ No encryption for face encodings
- ❌ No backup or data recovery mechanism

## Failure Cases

### Registration Failures
1. **Multiple faces in frame** → Only first face registered
2. **Poor lighting** → Encoding quality degrades
3. **Blurred motion** → Features become unreliable
4. **Partial occlusion** → Incomplete feature extraction

### Recognition Failures
1. **Lighting mismatch** → Training in bright light, testing in dim
2. **Distance change** → Registered close-up, recognized far away
3. **Expression extremes** → Smiling vs neutral vs angry
4. **Camera angle** → Front-facing registration, side-profile test

### Liveness Failures
1. **No blink within 1.5 seconds** → Times out
2. **Eyes not visible** → Glasses glare, hair, darkness
3. **Video replay** → System accepts pre-recorded blink
4. **Static image with moving eyes** → Cutout attack succeeds

## Non-Production Constraints

### ⚠️ Security Warnings
This system is **NOT suitable** for:
- Financial transactions or sensitive data access
- Unsupervised public deployment
- High-security environments (banks, airports)
- Legal/compliance requirements (GDPR face data)

### Safe Use Cases
This system **IS appropriate** for:
- Office attendance in trusted environments
- Classroom/school attendance tracking
- Gym/club check-in systems
- Proof-of-concept demonstrations
- Educational projects

### Required Mitigations for Production
1. **Replace Haar Cascades** → MTCNN, RetinaFace, or YOLO-Face
2. **Replace custom encoding** → FaceNet, ArcFace, or Dlib embeddings
3. **Upgrade liveness** → Active challenges (head turn, smile, speak)
4. **Add encryption** → Encrypt face encodings at rest
5. **Implement audit logs** → Track all access attempts
6. **Use database** → PostgreSQL with proper indexing
7. **Add authentication** → Admin login before data access
8. **Enforce GDPR compliance** → Consent, right to deletion

## Features
- ✅ Real-time face detection (Haar Cascade)
- ✅ Custom 128-dim face encoding
- ✅ Multi-sample registration (7 frames)
- ✅ Euclidean distance matching (threshold: 8.0)
- ✅ Punch-in/out attendance logging
- ✅ Basic blink-based liveness detection
- ✅ JSON + NumPy persistent storage
- ✅ 10-second cooldown between cycles

## Requirements
- Python 3.11+
- Webcam/Camera access
- OpenCV 4.x
- NumPy

## Setup

### 1. Clone repository
```bash
git clone https://github.com/shivam1342/face-auth-attendance.git
cd face-auth-attendance
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run application
```bash
python app.py
```

## Usage

### Keyboard Controls
- **`r`** - Register new face (captures 7 samples)
- **`i`** - Punch-in (with liveness check)
- **`o`** - Punch-out
- **`s`** - Show today's attendance summary
- **`l`** - List all registered faces
- **`q`** - Quit application

### Registration Flow
1. Press `r` when face is visible
2. Enter name in terminal
3. Hold still while system captures 7 samples
4. Green flash indicates successful capture
5. Face ID assigned and saved

### Attendance Flow
1. Press `i` to punch-in
2. System recognizes face
3. Blink detection activates
4. Blink naturally within 1.5 seconds
5. Attendance logged with timestamp
6. Press `o` to punch-out

## Project Structure
```
face-auth-attendance/
├── app.py                      # Main entry point - orchestrates all components
├── camera/
│   └── camera.py              # Webcam abstraction (frame capture)
├── face/
│   ├── detector.py            # Face detection (Haar Cascade)
│   ├── encoder.py             # 128-dim feature extraction
│   └── matcher.py             # Euclidean distance matching
├── attendance/
│   ├── attendance.py          # Punch-in/out logic
│   └── storage.py             # JSON/NumPy persistence
├── spoof/
│   └── liveness.py            # Blink-based anti-spoofing
├── utils/
│   ├── config.py              # System constants & thresholds
│   └── image_utils.py         # Preprocessing utilities
└── data/
    ├── registered_faces/
    │   ├── faces.json         # User metadata
    │   └── encodings.npy      # Face embeddings
    └── attendance_logs.json   # Punch-in/out records
```

## Technical Details

### Face Encoding (128 dimensions)
- **64 features**: Histogram of equalized grayscale (64 bins)
- **64 features**: 8×8 spatial grid intensity means
- Face resized to 128×128 for consistency
- Histogram equalization for lighting normalization

### Face Matching
- **Metric**: Euclidean distance
- **Threshold**: 8.0 (lower = stricter)
- **Confidence**: `1 - (distance / threshold)`
- Returns best match if within threshold

### Liveness Detection
- Detects eyes using Haar eye cascade
- Tracks eye visibility changes (visible → hidden → visible)
- Minimum blink duration: 0.15 seconds
- Timeout: 1.5 seconds
- **Warning**: Easily defeated by videos

### Storage Format
**faces.json**:
```json
[
  {
    "id": 0,
    "name": "shivam",
    "registered_at": "2026-01-29T17:52:00.123456"
  }
]
```

**encodings.npy**: Shape `(n_faces, 128)`, dtype `float32`

**attendance_logs.json**:
```json
[
  {
    "name": "shivam",
    "face_id": 0,
    "type": "punch_in",
    "timestamp": "2026-01-29T17:53:11.456789",
    "date": "2026-01-29",
    "time": "17:53:11"
  }
]
```

## Troubleshooting

### Face not detected?
- Ensure front-facing position (< 30° rotation)
- Improve lighting (>200 lux recommended)
- Move closer (minimum 30×30 pixel face size)
- Remove glasses/hats if causing issues

### Recognition failing?
- Check lighting consistency (register and test in similar conditions)
- Increase threshold in `utils/config.py` (try 10.0-12.0)
- Re-register face if appearance changed significantly
- Ensure only one face in frame during recognition

### Liveness timing out?
- System currently has blink detection enabled
- Blink naturally within 1.5 seconds
- Ensure eyes are visible (no glare/hair)
- Test in well-lit environment

### Data corruption?
```bash
# Windows
Remove-Item data\registered_faces\* -Force
Remove-Item data\attendance_logs.json -Force

# Linux/Mac
rm -rf data/registered_faces/*
rm data/attendance_logs.json
```

## Configuration

Edit `utils/config.py` to tune system behavior:
```python
FACE_MATCH_THRESHOLD = 8.0        # Lower = stricter matching
REGISTRATION_SAMPLES = 7          # Samples per registration
LIVENESS_TIMEOUT_SECONDS = 1.5    # Blink detection timeout
MIN_DETECTION_CONFIDENCE = 0.7    # Face detection threshold
```

## Contributing
Pull requests welcome! Please:
- Follow existing code structure
- Document design decisions in docstrings
- Test with multiple faces/lighting conditions
- Update README for new features

## License
MIT License

## Author
Shivam - [GitHub](https://github.com/shivam1342)
