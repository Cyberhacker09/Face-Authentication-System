import numpy as np
import cv2

def calculate_distance(p1, p2):
    """Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_ear(eye_points, landmarks):
    """
    Calculate Eye Aspect Ratio (EAR).
    eye_points: indices of the eye landmarks (e.g., [33, 160, 158, 133, 153, 144] for Mediapipe)
    landmarks: list of (x, y) coordinates
    """
    # Vertical landmarks
    A = calculate_distance(landmarks[eye_points[1]], landmarks[eye_points[5]])
    B = calculate_distance(landmarks[eye_points[2]], landmarks[eye_points[4]])
    
    # Horizontal landmarks
    C = calculate_distance(landmarks[eye_points[0]], landmarks[eye_points[3]])
    
    if C == 0: return 0.0
    
    ear = (A + B) / (2.0 * C)
    return ear

def get_mar(mouth_points, landmarks):
    """
    Calculate Mouth Aspect Ratio (MAR).
    mouth_points: indices for mouth (e.g., inner lips)
    """
    # Vertical
    A = calculate_distance(landmarks[mouth_points[1]], landmarks[mouth_points[7]])
    B = calculate_distance(landmarks[mouth_points[2]], landmarks[mouth_points[6]])
    C = calculate_distance(landmarks[mouth_points[3]], landmarks[mouth_points[5]])
    
    # Horizontal
    D = calculate_distance(landmarks[mouth_points[0]], landmarks[mouth_points[4]])
    
    if D == 0: return 0.0
    
    mar = (A + B + C) / (2.0 * D)
    return mar

def get_head_pose(landmarks, frame_width, frame_height):
    """
    Estimate Head Pose (Pitch, Yaw, Roll) using PnP.
    Uses standard 3D model points and 2D landmarks.
    """
    # 2D Image points from landmarks
    # Nose tip, Chin, Left Eye Left Corner, Right Eye Right Corner, Left Mouth Corner, Right Mouth Corner
    # Mediapipe indices: 1 (nose), 152 (chin), 33 (L eye), 263 (R eye), 61 (L mouth), 291 (R mouth)
    
    image_points = np.array([
        landmarks[1],    # Nose tip
        landmarks[152],  # Chin
        landmarks[33],   # Left eye left corner
        landmarks[263],  # Right eye right corner
        landmarks[61],   # Left Mouth corner
        landmarks[291]   # Right mouth corner
    ], dtype="double")

    # Standard 3D model points
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # Camera internals
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    if not success:
        return 0, 0, 0

    # Project a 3D point (0, 0, 1000.0) onto the image plane to visualize
    # (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    # Calculate Euler angles
    rmat, jac = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Q, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    # angles: (pitch, yaw, roll)
    pitch = angles[0] * 360
    yaw = angles[1] * 360
    roll = angles[2] * 360

    return pitch, yaw, roll
