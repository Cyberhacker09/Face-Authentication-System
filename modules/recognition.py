import numpy as np
try:
    from deepface import DeepFace
except ImportError:
    print("Warning: DeepFace not installed. Recognition will fail.")
    DeepFace = None

class FaceRecognizer:
    def __init__(self, match_threshold=0.4):
        # VGG-Face with Cosine Similarity usually uses threshold around 0.40
        self.match_threshold = match_threshold
        self.model_name = "VGG-Face"

    def encode(self, frame, bbox):
        """
        Generates embedding for the face using DeepFace.
        bbox is (x1, y1, x2, y2).
        """
        if DeepFace is None:
            return None

        x1, y1, x2, y2 = bbox
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            return None
            
        try:
            # represent returns a list of dicts
            embedding_objs = DeepFace.represent(
                img_path=face_img,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend="skip"
            )
            
            if embedding_objs:
                return np.array(embedding_objs[0]["embedding"])
        except Exception as e:
            print(f"Encoding error: {e}")
        
        return None

    def identify(self, embedding, db_embeddings, db_ids, db_names):
        """
        Compare embedding against database using Cosine Similarity.
        Returns (user_id, name, distance, confidence).
        """
        if not db_embeddings:
            return None, "Unknown", 1.0, 0.0
            
        if embedding is None:
             return None, "Unknown", 1.0, 0.0

        # DeepFace VGG-Face embeddings are lists/arrays.
        # Cosine Distance = 1 - Cosine Similarity
        # Cosine Similarity = (A . B) / (||A|| * ||B||)
        
        best_match_idx = -1
        min_dist = float('inf')
        
        # Normalize input embedding once
        a = embedding
        norm_a = np.linalg.norm(a)
        
        for i, db_emb in enumerate(db_embeddings):
            b = np.array(db_emb)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                dist = 1.0
            else:
                cos_sim = np.dot(a, b) / (norm_a * norm_b)
                dist = 1.0 - cos_sim # Cosine Distance
            
            if dist < min_dist:
                min_dist = dist
                best_match_idx = i

        if min_dist < self.match_threshold and best_match_idx != -1:
            # Confidence is inverse of distance, normalized roughly
            confidence = max(0.0, 1.0 - min_dist)
            return db_ids[best_match_idx], db_names[best_match_idx], min_dist, confidence
        
        return None, "Unknown", min_dist, 0.0