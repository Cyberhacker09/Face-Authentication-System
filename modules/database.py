import numpy as np
import json
import os

class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self.users = {}
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                try:
                    self.users = json.load(f)
                    # Convert list embeddings back to numpy arrays
                    for user_id, data in self.users.items():
                        data['embedding'] = np.array(data['embedding'])
                except json.JSONDecodeError:
                    self.users = {}
        else:
            self.users = {}

    def save(self):
        # Convert numpy arrays to lists for JSON serialization
        serializable_users = {}
        for user_id, data in self.users.items():
            serializable_users[user_id] = data.copy()
            if isinstance(data['embedding'], np.ndarray):
                serializable_users[user_id]['embedding'] = data['embedding'].tolist()
            else:
                serializable_users[user_id]['embedding'] = data['embedding']
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with open(self.db_path, 'w') as f:
            json.dump(serializable_users, f, indent=4)

    def add_user(self, name, embedding, metadata=None):
        import uuid
        user_id = str(uuid.uuid4())
        self.users[user_id] = {
            "name": name,
            "embedding": embedding,
            "metadata": metadata or {},
            "created_at": str(np.datetime64('now'))
        }
        self.save()
        return user_id

    def get_all_embeddings(self):
        ids = []
        embeddings = []
        names = []
        for user_id, data in self.users.items():
            ids.append(user_id)
            embeddings.append(data['embedding'])
            names.append(data['name'])
        return ids, names, embeddings
