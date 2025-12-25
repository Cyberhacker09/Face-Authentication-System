import cv2
import numpy as np
import os

class FaceProcessor:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        # Using Haar Cascade for speed as MediaPipe is unavailable
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            print("Error: Could not load Haar Cascade XML.")

    def process(self, frame):
        """
        Process the frame and return face bounding box.
        Note: Landmarks are not available with Haar Cascade.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        rects = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces_data = []
        for (x, y, w, h) in rects:
            bbox = (x, y, x + w, y + h) # x1, y1, x2, y2
            
            # Simulated landmarks (Center, Top, Bottom, Left, Right)
            # strictly for visual placeholder, not accurate
            cx, cy = x + w//2, y + h//2
            landmarks_px = [
                (cx, cy), # Approx nose
                (cx, y + h), # Chin
                (x + w//4, y + h//3), # Left Eye approx
                (x + 3*w//4, y + h//3), # Right Eye approx
                (x + w//4, y + 2*h//3), # Left Mouth approx
                (x + 3*w//4, y + 2*h//3) # Right Mouth approx
            ]
            
            faces_data.append({
                "landmarks": landmarks_px, 
                "landmarks_normalized": None,
                "bbox": bbox
            })
                
        return faces_data
    
    def draw_landmarks(self, frame, faces_data):
        """Draws the box only."""
        for face in faces_data:
            x1, y1, x2, y2 = face["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return frame
