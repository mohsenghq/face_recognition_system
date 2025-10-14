# embedding.py
import torch
import numpy as np
import cv2
from insightface.model_zoo import get_model

class FaceEmbedder:
    def __init__(self, model_name='arcface_r100_v1', device='cuda'):
        self.device = device
        self.model = get_model(model_name)
        self.model.prepare(ctx_id=0 if device == 'cuda' else -1)

    def get_embeddings(self, enhanced_faces):
        embeddings = []
        for face in enhanced_faces:
            face_resized = cv2.resize(face, (112, 112))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            emb = self.model.get_embedding(face_rgb)
            embeddings.append(emb)
        return np.array(embeddings)

