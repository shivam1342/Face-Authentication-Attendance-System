"""
Centralized configuration for thresholds, timeouts, and system constants.
Allows easy tuning without modifying core logic.
"""

# Face Detection
MIN_DETECTION_CONFIDENCE = 0.7
MIN_FACE_SIZE = (30, 30)

# Face Matching
FACE_MATCH_THRESHOLD = 8.0  # Lower is more strict

# Face Registration
REGISTRATION_SAMPLES = 7
REGISTRATION_FRAME_INTERVAL = 8

# Liveness Detection
LIVENESS_REQUIRED_BLINKS = 0  # 0 = disabled
LIVENESS_TIMEOUT_SECONDS = 5
EYE_ASPECT_RATIO_THRESHOLD = 0.21

# Camera
DEFAULT_CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Storage Paths
DATA_DIR = "data"
FACES_DIR = "data/registered_faces"
FACES_JSON = "data/registered_faces/faces.json"
ENCODINGS_FILE = "data/registered_faces/encodings.npy"
ATTENDANCE_LOG = "data/attendance_logs.json"

# Haar Cascade Path
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"

