import os
import logging

# Suppress TensorFlow and Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import cv2
import time
import numpy as np
import threading
from collections import deque

# Config
import config

# Modules
from modules.camera import Camera
from modules.detection import FaceProcessor
from modules.tracker import CentroidTracker
from modules.quality import QualityChecker
from modules.liveness import LivenessDetector
from modules.recognition import FaceRecognizer
from modules.analysis import FaceAnalyzer
from modules.database import Database
from modules.ui import UI
from modules.geometry import calculate_distance

def main():
    print("Initializing System...")
    
    # 1. Initialize Modules
    cam = Camera(config.CAMERA_ID, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
    detector = FaceProcessor(config.MIN_DETECTION_CONFIDENCE, config.MIN_TRACKING_CONFIDENCE)
    tracker = CentroidTracker(max_disappeared=30)
    quality_checker = QualityChecker(
        blur_threshold=config.BLUR_THRESHOLD,
        min_brightness=config.MIN_BRIGHTNESS,
        max_brightness=config.MAX_BRIGHTNESS,
        max_yaw=config.MAX_YAW_ANGLE,
        max_pitch=config.MAX_PITCH_ANGLE,
        min_face_width=config.MIN_FACE_WIDTH_PX
    )
    liveness_detector = LivenessDetector(
        ear_thresh=config.EAR_THRESHOLD,
        mar_thresh=config.MAR_THRESHOLD,
        head_turn_thresh=config.HEAD_TURN_THRESHOLD,
        blink_consec_frames=config.BLINK_CONSEC_FRAMES
    )
    recognizer = FaceRecognizer(config.MATCH_THRESHOLD)
    analyzer = FaceAnalyzer()
    db = Database(config.DB_PATH)
    ui = UI()

    # Load known faces
    known_ids, known_names, known_embeddings = db.get_all_embeddings()
    print(f"Loaded {len(known_ids)} users from database.")

    # State Management for Tracked IDs
    # format: { track_id: { 'name': str, 'verified': bool, 'liveness_status': str, 'attributes': {}, 'challenge': str, 'start_time': float } }
    track_states = {}
    
    print("System Ready. Press 'q' to quit. Press 'r' to register the current face.")
    
    frame_count = 0
    fps_start_time = time.time()
    
    register_mode = False
    register_name_buffer = ""

    try:
        while True:
            if cam is None:
                break
            frame = cam.read()
            if frame is None:
                continue

            frame_count += 1
            h, w, _ = frame.shape

            # 2. Detect Faces
            faces_data = detector.process(frame)
            
            # 3. Prepare Rects for Tracker
            rects = []
            for face in faces_data:
                rects.append(face['bbox']) # (x1, y1, x2, y2)
            
            # 4. Update Tracker
            objects = tracker.update(rects)
            
            # 5. Map Track IDs to Face Data
            # We match tracker centroid to face bbox center
            tracked_faces = []
            for (objectID, centroid) in objects.items():
                # Find closest face_data
                best_match = None
                min_dist = float('inf')
                
                for face in faces_data:
                    bbox = face['bbox']
                    cx = (bbox[0] + bbox[2]) // 2
                    cy = (bbox[1] + bbox[3]) // 2
                    dist = calculate_distance(centroid, (cx, cy))
                    if dist < 50: # Threshold to associate
                        if dist < min_dist:
                            min_dist = dist
                            best_match = face
                
                if best_match:
                    tracked_faces.append((objectID, best_match))

            # Clean up old states
            active_ids = objects.keys()
            track_states = {k: v for k, v in track_states.items() if k in active_ids}

            # 6. Process Each Tracked Face
            for track_id, face_data in tracked_faces:
                bbox = face_data['bbox']
                landmarks = face_data['landmarks']
                
                # Initialize state if new
                if track_id not in track_states:
                    track_states[track_id] = {
                        'name': "Unknown",
                        'verified': False,
                        'liveness_status': "PENDING",
                        'attributes': {},
                        'challenge': None,
                        'quality_ok': False,
                        'embedding': None,
                        'welcome_printed': False
                    }
                
                state = track_states[track_id]

                # A. Quality Check
                quality_ok, quality_details = quality_checker.evaluate(frame, face_data)
                state['quality_ok'] = quality_ok
                
                # Draw Box (Red if bad quality/unknown, Green if verified)
                color = "red"
                if state['verified']: color = "green"
                elif state['liveness_status'] == "PASSED": color = "yellow"
                
                ui.draw_box(frame, bbox, color, label=f"ID: {track_id}")

                # B. Liveness Logic (Only if quality is OK and not yet passed)
                if quality_ok and state['liveness_status'] != "PASSED":
                    if state['challenge'] is None:
                        # Start a challenge
                        state['challenge'] = liveness_detector.start_new_challenge()
                    
                    # Update Liveness Detector with current challenge
                    liveness_detector.current_challenge = state['challenge']
                    
                    # Pass landmarks to detector (we need to handle the state inside detector better for multiple faces? 
                    # Actually LivenessDetector class stores state. If multiple faces, we need multiple instances or pass state in.
                    # Simplified: We only auth one person at a time effectively, or reset detector.
                    # BETTER: Instantiate LivenessDetector per track_id? 
                    # For now, let's assume single-user interaction focus or reset for simplicity.
                    # The current LivenessDetector holds state (counters). 
                    # We will re-instantiate or use a dictionary in LivenessDetector.
                    # For this prototype, we'll assume the FOCUSED user (closest/largest) drives the main LivenessDetector,
                    # or just create a new one for each track in `track_states`.
                    
                    if 'liveness_obj' not in state:
                        state['liveness_obj'] = LivenessDetector(
                            ear_thresh=config.EAR_THRESHOLD, 
                            mar_thresh=config.MAR_THRESHOLD, 
                            head_turn_thresh=config.HEAD_TURN_THRESHOLD, 
                            blink_consec_frames=config.BLINK_CONSEC_FRAMES
                        )
                        state['liveness_obj'].start_new_challenge()
                        state['challenge'] = state['liveness_obj'].current_challenge
                    
                    success, msg = state['liveness_obj'].process(landmarks, w, h)
                    
                    ui.draw_text(frame, f"Liveness: {state['challenge']} ({msg})", (bbox[0], bbox[1]-30), "yellow")
                    
                    if success:
                        state['liveness_status'] = "PASSED"
                        state['challenge'] = "PASSED"
                
                elif not quality_ok:
                    # Show why
                    reasons = []
                    checks = quality_details['checks']
                    if not checks['clear']: reasons.append("BLUR")
                    if not checks['lit']: reasons.append("DARK")
                    if not checks['frontal']: reasons.append("POSE")
                    if not checks['size']: reasons.append("FAR")
                    ui.draw_text(frame, f"Quality Fail: {','.join(reasons)}", (bbox[0], bbox[3]+20), "red")

                # C. Recognition & Analysis (Once Liveness Passed)
                if state['liveness_status'] == "PASSED" and not state['verified']:
                    
                    # 1. Identify
                    if state['embedding'] is None:
                         # Encode
                         emb = recognizer.encode(frame, bbox)
                         if emb is not None:
                             state['embedding'] = emb
                             # Match
                             uid, name, dist, conf = recognizer.identify(emb, known_embeddings, known_ids, known_names)
                             state['name'] = name
                             state['conf'] = conf
                             if name != "Unknown":
                                 state['verified'] = True
                    
                    # 2. Analyze (Attribute) - Run once
                    if not state['attributes']:
                         attrs = analyzer.analyze(frame, bbox)
                         state['attributes'] = attrs

                # D. Display Info
                if state['verified']:
                    # Welcome Message on UI
                    welcome_msg = f"WELCOME {state['name'].upper()}"
                    ui.draw_text(frame, welcome_msg, (bbox[0], bbox[1] - 50), "green", scale=1.0, thickness=2)

                    # Console Welcome (once per track)
                    if not state.get('welcome_printed', False):
                        print(f"Welcome, {state['name']}!")
                        state['welcome_printed'] = True

                    info = [
                        f"Name: {state['name']}",
                        f"Conf: {state.get('conf', 0):.2f}",
                        f"Age: {state['attributes'].get('age', '?')}",
                        f"Gender: {state['attributes'].get('gender', '?')}",
                        f"Emotion: {state['attributes'].get('emotion', '?')}"
                    ]
                    for i, line in enumerate(info):
                        ui.draw_text(frame, line, (bbox[2]+10, bbox[1] + 20 + (i*20)), "green")
                
                # E. Registration Hook
                if register_mode and state['quality_ok']:
                    # Auto capture if one face
                    if len(tracked_faces) == 1:
                        # Pause updates
                        register_mode = False 
                        
                        # In CLI, input() blocks the loop. 
                        # We need to capture input without freezing, or just freeze temporarily.
                        # Let's freeze.
                        cv2.imshow("Facial Auth System", frame)
                        cv2.waitKey(1)
                        print("\n=== REGISTRATION ===")
                        name = input("Enter name for new user: ")
                        if name:
                            emb = recognizer.encode(frame, bbox)
                            if emb is not None:
                                db.add_user(name, emb, state['attributes'])
                                print(f"User {name} added successfully.")
                                # Reload DB
                                known_ids, known_names, known_embeddings = db.get_all_embeddings()
                            else:
                                print("Failed to encode face. Try again.")
                        else:
                            print("Registration cancelled.")

            # 7. Global UI
            fps = frame_count / (time.time() - fps_start_time)
            stats = {
                "FPS": f"{fps:.1f}",
                "Faces": len(tracked_faces),
                "Mode": "REGISTER (Press 'r')" if not register_mode else "CAPTURING...",
            }
            ui.draw_dashboard(frame, stats)
            
            # Show Mesh (Optional, good for debug)
            # detector.draw_landmarks(frame, faces_data) 

            cv2.imshow("Facial Auth System", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                print("Switching to Registration Mode...")
                register_mode = True

    finally:
        if cam is not None:
            cam.stop()
        cv2.destroyAllWindows()
        print("System Shutdown.")

if __name__ == "__main__":
    main()
