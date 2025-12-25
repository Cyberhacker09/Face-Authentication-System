import cv2
import numpy as np

class UI:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0)
        }

    def draw_box(self, frame, bbox, color_name="green", label=None):
        x1, y1, x2, y2 = bbox
        color = self.colors.get(color_name, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if label:
            self.draw_text(frame, label, (x1, y1 - 10), color_name=color_name)

    def draw_text(self, frame, text, pos, color_name="green", scale=0.6, thickness=1):
        color = self.colors.get(color_name, (0, 255, 0))
        # Draw background for text readability
        (w, h), _ = cv2.getTextSize(text, self.font, scale, thickness)
        x, y = pos
        cv2.rectangle(frame, (x, y - h - 5), (x + w, y + 5), self.colors["black"], -1)
        cv2.putText(frame, text, pos, self.font, scale, color, thickness)

    def draw_dashboard(self, frame, stats):
        """
        Draws a dashboard on the right side or top.
        stats: dict of key-values
        """
        y = 30
        x = 20
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 300), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        self.draw_text(frame, "SYSTEM STATUS", (x, y), "white", 0.7, 2)
        y += 30
        
        for k, v in stats.items():
            text = f"{k}: {v}"
            color = "white"
            if "DENY" in str(v) or "FAIL" in str(v): color = "red"
            if "ALLOW" in str(v) or "PASS" in str(v): color = "green"
            self.draw_text(frame, text, (x, y), color)
            y += 25

    def draw_liveness_challenge(self, frame, challenge_name):
        h, w, _ = frame.shape
        text = f"ACTION REQUIRED: {challenge_name}"
        (tw, th), _ = cv2.getTextSize(text, self.font, 1.2, 3)
        x = (w - tw) // 2
        y = h - 50
        
        cv2.rectangle(frame, (x - 20, y - th - 20), (x + tw + 20, y + 20), (0,0,255), -1)
        cv2.putText(frame, text, (x, y), self.font, 1.2, (255, 255, 255), 3)
