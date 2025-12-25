import os

# Suppress AI Framework Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# System Configuration

# Camera
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Face Detection
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Quality Control Thresholds
MIN_FACE_WIDTH_PX = 80
MAX_YAW_ANGLE = 25  # degrees
MAX_PITCH_ANGLE = 25 # degrees
BLUR_THRESHOLD = 50 # Laplacian variance (higher is clearer)
MIN_BRIGHTNESS = 70 # 0-255
MAX_BRIGHTNESS = 220

# Liveness (Active)
# EAR = Eye Aspect Ratio, MAR = Mouth Aspect Ratio
EAR_THRESHOLD = 0.22 # Below this is closed
BLINK_CONSEC_FRAMES = 2 # Number of frames to confirm blink
MAR_THRESHOLD = 0.5 # Above this is open mouth
HEAD_TURN_THRESHOLD = 20 # degrees

# Recognition
MATCH_THRESHOLD = 0.5 # Lower is stricter (Euclidean distance)

# Paths
DB_PATH = "database/users.json"
LOG_PATH = "auth.log"
