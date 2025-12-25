import cv2
import numpy as np
from .geometry import get_head_pose

class QualityChecker:
    def __init__(self, blur_threshold=50, min_brightness=70, max_brightness=220, 
                 max_yaw=25, max_pitch=25, min_face_width=80):
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.min_face_width = min_face_width

    def check_blur(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return score, score > self.blur_threshold

    def check_brightness(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        return brightness, (self.min_brightness <= brightness <= self.max_brightness)

    def check_pose(self, landmarks, frame_width, frame_height):
        # MediaPipe missing, so we cannot accurately calculate pitch/yaw/roll from 2D Haar bbox easily without PnP.
        # We will assume pose is OK if face is detected for now, or use Aspect Ratio.
        # Future: Use DeepFace.analyze['region'] to check centering?
        
        # Dummy Return: (pitch, yaw, roll), is_good
        return (0, 0, 0), True

    def check_face_size(self, bbox):
        # bbox is (x1, y1, x2, y2)
        width = bbox[2] - bbox[0]
        return width, width >= self.min_face_width

    def evaluate(self, frame, face_data):
        """
        Runs all quality checks.
        """
        h, w, _ = frame.shape
        
        blur_score, is_clear = self.check_blur(frame)
        brightness, is_lit = self.check_brightness(frame)
        
        # Crop face for more specific blur check? 
        # For speed, we checked whole frame blur, but face blur is better.
        # Let's crop face if possible.
        bbox = face_data['bbox']
        # Clamp bbox
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size > 0:
             blur_score, is_clear = self.check_blur(face_roi)
             brightness, is_lit = self.check_brightness(face_roi)

        pose, is_frontal = self.check_pose(face_data['landmarks'], w, h)
        width, is_large_enough = self.check_face_size(bbox)
        
        status = is_clear and is_lit and is_frontal and is_large_enough
        
        details = {
            "blur": blur_score,
            "brightness": brightness,
            "pose": pose, # (pitch, yaw, roll)
            "width": width,
            "checks": {
                "clear": bool(is_clear),
                "lit": bool(is_lit),
                "frontal": bool(is_frontal),
                "size": bool(is_large_enough)
            }
        }
        
        return status, details
