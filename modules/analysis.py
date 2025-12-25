try:
    from deepface import DeepFace
except ImportError:
    print("Warning: DeepFace not installed. Attribute analysis will fail.")
    DeepFace = None

class FaceAnalyzer:
    def __init__(self):
        pass

    def analyze(self, frame, bbox):
        """
        Predict Age, Gender, Emotion.
        """
        if DeepFace is None:
            return {}

        # Crop face
        x1, y1, x2, y2 = bbox
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            return {}

        try:
            # DeepFace expects RGB usually, but handles BGR if backend is opencv? 
            # DeepFace.analyze loads image from path or numpy.
            # enforce_detection=False because we already cropped it.
            results = DeepFace.analyze(
                img_path=face_img, 
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False,
                silent=True,
                detector_backend='skip' # Important for speed
            )
            
            # DeepFace returns a list of dicts (for multiple faces) or single dict
            if isinstance(results, list):
                res = results[0]
            else:
                res = results
                
            return {
                "age": res.get("age"),
                "gender": res.get("dominant_gender"),
                "emotion": res.get("dominant_emotion"),
                "emotion_score": res.get("emotion") # dict of scores
            }
        except Exception as e:
            # print(f"Analysis error: {e}")
            return {}
