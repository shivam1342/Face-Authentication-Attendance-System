# Basic Heuristic-Based Liveness Detection

## ⚠️ Important Disclaimer

This is **NOT a security-grade anti-spoofing system**. This is a basic heuristic check suitable for controlled attendance scenarios like office/classroom environments.

### Can be defeated by:
- Pre-recorded videos of someone blinking
- Sophisticated presentation attacks
- High-quality printed photos with cutout eyes

### Suitable for:
- Low-security attendance tracking
- Controlled environments with supervision
- Basic prevention against simple photo spoofing
- Scenarios where social deterrence is sufficient

---

## How It Works

### Blink Detection Using Eye Aspect Ratio (EAR)

1. **Eye Detection**: Uses Haar Cascade to detect eyes within the face region
2. **EAR Calculation**: Measures eye height-to-width ratio (simplified proxy for true EAR)
3. **Blink Detection**: Tracks when eyes close (low EAR) then open again
4. **Verification**: Requires at least 1 natural blink within 3 seconds

### Algorithm Flow

```
User faces camera
  ↓
Detect face region
  ↓
Detect eyes within face
  ↓
Calculate Eye Aspect Ratio (EAR)
  ↓
Track consecutive frames with closed eyes
  ↓
When eyes reopen after closure → Count as blink
  ↓
If blink count ≥ threshold → Liveness verified
  ↓
If timeout (3s) without blink → Verification failed
```

---

## Usage

### Basic Test

```bash
python test_liveness.py
```

1. Look at camera
2. Press `R` to start liveness check
3. Blink naturally
4. System verifies within 3 seconds

### Integration Example

```python
from spoof.liveness import LivenessDetector
from face.detector import FaceDetector

detector = FaceDetector()
liveness = LivenessDetector()

# Reset before starting new check
liveness.reset()

# In video loop:
while True:
    frame = get_camera_frame()
    faces = detector.detect(frame)
    
    if faces:
        result = liveness.verify_liveness(frame, faces[0]['bbox'])
        
        if result['is_live'] is True:
            print("✓ Liveness verified!")
            break
        elif result['is_live'] is False:
            print("✗ Liveness check failed")
            break
        else:
            # Still checking, show message
            print(result['message'])
```

---

## Configuration Parameters

In `spoof/liveness.py`:

```python
self.blink_threshold = 0.25      # EAR threshold (lower = stricter)
self.consecutive_frames = 2       # Frames to confirm eye closure
self.required_blinks = 1          # Minimum blinks needed
self.max_check_duration = 3.0     # Seconds before timeout
```

### Tuning Tips:

- **Lower `blink_threshold`**: More sensitive, may miss blinks with small eyes
- **Higher `required_blinks`**: More secure but slower (2-3 blinks takes longer)
- **Longer `max_check_duration`**: Give users more time (useful for nervous users)

---

## Limitations

### Technical Limitations:
1. **Haar Cascade eye detection** is not perfect
   - Can miss eyes in low light
   - Fails with glasses/sunglasses sometimes
   - Sensitive to head angles

2. **Simplified EAR calculation**
   - Uses height/width ratio instead of true landmark-based EAR
   - Less accurate than dlib/MediaPipe facial landmarks
   - May trigger false positives with eye movement

3. **No depth perception**
   - Cannot distinguish between real face and video screen
   - No 3D structure analysis

### Practical Limitations:
- Users with glasses may have difficulty
- Very bright or dim lighting affects detection
- Requires frontal face view
- Cannot detect sophisticated spoofing

---

## When to Use This

✅ **Good for:**
- Office attendance systems with supervision
- School/university check-ins
- Gym/club access logging
- Low-stakes verification where social deterrence is enough

❌ **Not suitable for:**
- Financial transactions
- High-security access control
- Remote unsupervised authentication
- Critical infrastructure access

---

## Improvements (If Needed Later)

If you need stronger liveness detection:

1. **Add head movement check**
   - Require user to turn head left/right
   - Track face position changes

2. **Use facial landmarks** (dlib or MediaPipe)
   - More accurate EAR calculation
   - Better eye detection

3. **Texture analysis**
   - Check for screen reflections
   - Detect printed photo artifacts

4. **Use commercial API**
   - FaceTec, iProov, or similar
   - True 3D liveness with depth sensors

---

## Testing Results

Expected behavior:
- **True Positive**: User blinks naturally → Verified within 1-2 seconds
- **False Negative**: User doesn't blink or looks away → Timeout after 3 seconds
- **Limitations**: Photo spoofing may pass if photo shows eyes clearly (no blink required in current version)

---

## Implementation Notes

**Why blink detection?**
- Simple to implement with OpenCV
- Works with existing Haar Cascade setup
- Natural action (users blink unconsciously)
- Better UX than requiring head movement

**Why not use MediaPipe/dlib?**
- Adds heavy dependencies
- We already switched from MediaPipe for simplicity
- Haar Cascade sufficient for controlled environments

**Trade-offs:**
- Security ↔ Simplicity: Chose simplicity
- Accuracy ↔ Dependencies: Minimized dependencies
- Speed ↔ Robustness: Fast checks (3s timeout)
