import random
import time

class LivenessDetector:
    def __init__(self, ear_thresh=None, mar_thresh=None, head_turn_thresh=None, blink_consec_frames=None):
        # Arguments are kept for compatibility with main.py calls, but ignored.
        
        # New Challenges suitable for Bounding Box only
        self.challenges = ["MOVE_CLOSER", "MOVE_AWAY", "MOVE_LEFT", "MOVE_RIGHT"]
        self.current_challenge = None
        self.challenge_start_time = 0
        self.challenge_timeout = 5.0 # seconds
        self.challenge_completed = False
        
        self.initial_bbox = None # (x1, y1, x2, y2)
        self.initial_center = None
        self.initial_width = 0

    def start_new_challenge(self):
        self.current_challenge = random.choice(self.challenges)
        self.challenge_start_time = time.time()
        self.challenge_completed = False
        self.initial_bbox = None
        return self.current_challenge

    def process(self, landmarks, frame_w, frame_h):
        # 'landmarks' here is the dummy list from detection.py
        # Point 0 is Center (cx, cy).
        # Point 1 is Chin (cx, y+h).
        # Point 2/3 are eye/mouth approx which encode width.
        
        if not self.current_challenge:
            return False, "No active challenge"

        if time.time() - self.challenge_start_time > self.challenge_timeout:
            return False, "TIMEOUT"
            
        # Parse Geometry from Dummy Landmarks
        try:
            # Center
            cx, cy = landmarks[0]
            # Width approx (landmarks[2] is left eye, [3] is right eye)
            # In detection.py: 
            # lx = x + w//4
            # rx = x + 3*w//4
            # rx - lx = w/2. So Width = (rx - lx) * 2
            lx, _ = landmarks[2]
            rx, _ = landmarks[3]
            width = (rx - lx) * 2
        except:
            return False, "Tracking Error"

        if self.initial_bbox is None:
            self.initial_center = (cx, cy)
            self.initial_width = width
            self.initial_bbox = True # Flag that we started
            return False, "Keep Moving..."

        dx = cx - self.initial_center[0]
        dw = width - self.initial_width
        
        success = False
        threshold_move = frame_w * 0.05 # 5% screen width
        threshold_zoom = self.initial_width * 0.2 # 20% size change

        if self.current_challenge == "MOVE_LEFT":
            # Moving left means x decreases (assuming camera is mirrored or standard)
            # Usually detection x moves to 0 (Left).
            if dx < -threshold_move: success = True
            
        elif self.current_challenge == "MOVE_RIGHT":
             if dx > threshold_move: success = True
             
        elif self.current_challenge == "MOVE_CLOSER":
            # Width increases
            if dw > threshold_zoom: success = True
            
        elif self.current_challenge == "MOVE_AWAY":
            # Width decreases
            if dw < -threshold_zoom: success = True

        if success:
            self.challenge_completed = True
            return True, "PASSED"
        
        return False, "WAITING..."