# embedding.py
import torch
import numpy as np
import cv2
from insightface.app import FaceAnalysis

class FaceEmbedder:
    def __init__(self, model_name='buffalo_l', device='cuda'):
        self.device = device
        # Use FaceAnalysis instead of get_model
        self.model = FaceAnalysis(name=model_name)
        self.model.prepare(ctx_id=0 if device == 'cuda' else -1)

    def get_embeddings(self, enhanced_faces):
        embeddings = []
        for face in enhanced_faces:
            face_resized = cv2.resize(face, (112, 112))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            # Use get method instead of get_embedding
            faces = self.model.get(face_rgb)
            if faces:
                emb = faces[0].embedding
                embeddings.append(emb)
            else:
                # If no face detected, create zero embedding
                embeddings.append(np.zeros(512))
        return np.array(embeddings)

